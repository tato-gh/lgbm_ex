defmodule LgbmEx.Model do
  @moduledoc """
  Model struct
  """

  alias LgbmEx.NIFAPI
  alias LgbmEx.LightGBM
  alias LgbmEx.ModelFile

  defstruct [
    :workdir,
    :name,
    :files,
    :parameters,
    :ref,
    :num_iterations,
    :learning_steps,
    :used_parameters,
    :num_classes,
    :num_features,
    :feature_importance_split,
    :feature_importance_gain
  ]

  @doc """
  Returns %Model{} after train
  """
  def train(name, {csv_train, csv_val}, parameters) do
    model =
      init_model(name)
      |> prepare_train(parameters, validation: true)

    ModelFile.write!(model.files.train, csv_train)
    ModelFile.write!(model.files.validation, csv_val)
    ModelFile.write_parameters!(model.files.parameter, model.parameters)

    LightGBM.train(model) |> merge_train_result()
  end

  def train(name, {csv_train}, parameters) do
    model =
      init_model(name)
      |> prepare_train(parameters, validation: true)

    ModelFile.write!(model.files.train, csv_train)
    ModelFile.maybe_make_hard_link(model.files.train, model.files.validation)
    ModelFile.write_parameters!(model.files.parameter, model.parameters)

    LightGBM.train(model) |> merge_train_result()
  end

  def train(name, csv_train, parameters) do
    model =
      init_model(name)
      |> prepare_train(parameters, validation: false)

    ModelFile.write!(model.files.train, csv_train)
    ModelFile.write_parameters!(model.files.parameter, model.parameters)

    LightGBM.train(model) |> merge_train_result()
  end

  def predict(model, %Explorer.DataFrame{} = df) do
    predict(model, LgbmEx.DataFrame.to_x(df, Keyword.get(model.parameters, :x_names)))
  end

  def predict(model, list) when is_list(list) do
    LightGBM.predict(model, list)
  end

  def load(name) do
    model = init_model(name)
    parameters = ModelFile.read_parameters!(model.files.parameter)
    model = prepare_train(model, parameters)
    {:ok, ref} = NIFAPI.create_reference(model)

    Map.put(model, :ref, ref) |> merge_train_result()
  end

  def refit(model, parameters) do
    parameters = Keyword.merge(model.parameters, parameters)
    model = prepare_train(model, parameters)
    ModelFile.write_parameters!(model.files.parameter, model.parameters)

    LightGBM.train(model) |> merge_train_result()
  end

  def copy(model, name) do
    dest_dir = Path.join(get_workdir(), name)

    File.mkdir_p(dest_dir)
    Enum.each(model.files, fn {_key, src_file} ->
      dest_file = Path.join(dest_dir, Path.basename(src_file))
      File.cp(src_file, dest_file, on_conflict: fn _, _ -> true end)
    end)

    model
    |> Map.put(:name, name)
    |> put_files()
  end

  def zip(model) do
    dir = Path.join(model.workdir, model.name)
    model_files = Enum.map(model.files, & Path.basename(elem(&1, 1))) |> MapSet.new()
    existing_files = File.ls!(dir) |> MapSet.new()

    files =
      MapSet.intersection(model_files, existing_files)
      |> MapSet.to_list()
      |> Enum.map(& String.to_charlist/1)

    tmp_dir = System.tmp_dir!()
    zip_name = Path.join(tmp_dir, model.name <> ".zip") |> String.to_charlist()

    :zip.create(zip_name, files, cwd: dir)
  end

  def unzip(path, name) do
    path = String.to_charlist(path)
    workdir = get_workdir()
    dir = Path.join(workdir, name) |> String.to_charlist()
    :zip.extract(path, cwd: dir)
    load(name)
  end

  defp init_model(name) do
    workdir = get_workdir()

    %__MODULE__{workdir: workdir, name: name}
    |> put_files()
  end

  defp put_files(model) do
    model_dir = Path.join(model.workdir, model.name)
    File.mkdir_p(model_dir)

    Map.put(model, :files, %{
      model: Path.join(model_dir, "model.txt"),
      train: Path.join(model_dir, "train.csv"),
      validation: Path.join(model_dir, "validation.csv"),
      parameter: Path.join(model_dir, "parameter.txt"),
      train_log: Path.join(model_dir, "train_log.txt")
    })
  end

  defp prepare_train(model, parameters, options \\ []) do
    model
    |> clear_ref()
    |> put_parameters()
    |> maybe_with_validation(Keyword.get(options, :validation))
    |> merge_parameters(parameters)
  end

  defp get_workdir do
    Application.fetch_env!(:lgbm_ex, :workdir)
  end

  defp clear_ref(model), do: Map.put(model, :ref, nil)

  # defp clear_ref(%{ref: nil} = model), do: model
  #
  # defp clear_ref(%{ref: ref} = model) do
  #   NIFAPI.call(:booster_free, ref)
  #   # => Segmentation fault
  #   Map.put(model, :ref, nil)
  # end

  defp put_parameters(model) do
    Map.put(model, :parameters, [
      task: "train",
      data: model.files.train,
      output_model: model.files.model,
      label_column: 0,
      saved_feature_importance_type: 1
    ])
  end

  defp merge_parameters(model, parameters) do
    custom_parameters =
      Keyword.drop(parameters, [:task, :data, :output_model, :label_column])

    Map.update!(model, :parameters, & Keyword.merge(&1, custom_parameters))
  end

  defp maybe_with_validation(model, true) do
    Map.update!(model, :parameters, & Keyword.merge(&1, [
      valid_data: model.files.validation
    ]))
  end

  defp maybe_with_validation(model, false) do
    Map.update!(model, :parameters, & Keyword.merge(&1, [
      valid_data: nil
    ]))
  end

  defp maybe_with_validation(model, nil), do: model

  defp merge_train_result(%{ref: ref} = model) do
    {num_iterations, learning_steps} =
      ModelFile.parse_train_log(model.files.train_log, Keyword.get(model.parameters, :metric))

    [feature_importance_split, feature_importance_gain] =
      NIFAPI.call(:booster_feature_importance, ref)

    Map.merge(model, %{
      num_iterations: num_iterations,
      learning_steps: learning_steps,
      used_parameters: NIFAPI.call(:booster_get_loaded_param, ref) |> Jason.decode!(),
      num_classes: NIFAPI.call(:booster_get_num_classes, ref),
      num_features: NIFAPI.call(:booster_get_num_features, ref),
      feature_importance_split: feature_importance_split,
      feature_importance_gain: feature_importance_gain
    })
  end
end
