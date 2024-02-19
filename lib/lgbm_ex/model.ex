defmodule LgbmEx.Model do
  @moduledoc """
  Model struct
  """

  alias LgbmEx.NIFAPI
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
    :feature_importance
  ]

  @first_name "cache"

  @doc """
  Returns %Model{} with cache directory.
  """
  def new_model(workdir) do
    %__MODULE__{workdir: workdir, name: @first_name}
    |> put_files()
  end

  @doc """
  Returns %Model{} setup before fit.
  """
  def setup_model(model, parameters, options \\ []) do
    model
    |> clear_ref()
    |> put_parameters()
    |> merge_parameters(parameters)
    |> maybe_with_validation(Keyword.get(options, :validation))
  end

  @doc """
  Returns %Model{} loaded from given workdir and name.
  """
  def load_model(workdir, name) do
    model =
      %__MODULE__{workdir: workdir, name: name}
      |> put_files()
      |> load_parameters()

    {:ok, ref} = NIFAPI.create_reference(model)

    Map.put(model, :ref, ref)
    |> complement_model_attrs()
  end

  @doc """
  Returns %Model{} set attrs after learning.
  """
  def complement_model_attrs(%{ref: ref} = model) do
    {num_iterations, learning_steps} =
      ModelFile.parse_train_log(model.files.train_log, Keyword.get(model.parameters, :metric))

    Map.merge(model, %{
      num_iterations: num_iterations,
      learning_steps: learning_steps,
      used_parameters: NIFAPI.call(:booster_get_loaded_param, ref) |> Jason.decode!(),
      num_classes: NIFAPI.call(:booster_get_num_classes, ref),
      num_features: NIFAPI.call(:booster_get_num_features, ref),
      feature_importance: NIFAPI.call(:booster_feature_importance, ref)
    })
  end

  @doc """
  Copy model to given name directory.
  """
  def copy_model(model, name) do
    dest_dir = Path.join(model.workdir, name)

    File.mkdir_p(dest_dir)
    Enum.each(model.files, fn {_key, src_file} ->
      dest_file = Path.join(dest_dir, Path.basename(src_file))
      File.cp(src_file, dest_file, on_conflict: fn _, _ -> true end)
    end)

    model
    |> Map.put(:name, name)
    |> put_files()
  end

  @doc """
  Copy model to sub directory with given name. Data files are copied by hard link.
  """
  def copy_model_as_sub(model, name) do
    sub_workdir = Path.join(model.workdir, model.name)
    dest_dir = Path.join(sub_workdir, name)

    File.mkdir_p(dest_dir)
    Enum.each(model.files, fn {_key, src_file} ->
      dest_file = Path.join(dest_dir, Path.basename(src_file))

      String.ends_with?(src_file, ".csv")
      |> if do
        File.rm(dest_file)
        maybe_make_hard_link(src_file, dest_file)
      else
        # not shared files, like model.txt
        File.cp(src_file, dest_file, on_conflict: fn _, _ -> true end)
      end
    end)

    model
    |> Map.merge(%{workdir: sub_workdir, name: name})
    |> put_files()
  end

  @doc """
  Returns %Model{} merged given parameters.
  """
  def merge_parameters(model, parameters) do
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

  defp maybe_with_validation(model, _), do: model

  defp put_files(model) do
    dir = Path.join(model.workdir, model.name)

    if model.name == @first_name do
      File.rm_rf(dir)
      File.mkdir_p(dir)
    end

    Map.put(model, :files, %{
      model: Path.join(dir, "model.txt"),
      train: Path.join(dir, "train.csv"),
      validation: Path.join(dir, "validation.csv"),
      parameter: Path.join(dir, "paramter.txt"),
      train_log: Path.join(dir, "train_log.txt")
    })
  end

  defp put_parameters(model) do
    Map.put(model, :parameters, [
      task: "train",
      data: model.files.train,
      output_model: model.files.model,
      label_column: 0,
      saved_feature_importance_type: 1
    ])
  end

  defp load_parameters(model) do
    parameters = ModelFile.read_parameters(model.files.parameter)

    model
    |> put_parameters()
    |> merge_parameters(parameters)
  end

  defp maybe_make_hard_link(src_file, dest_file) do
    File.ln(src_file, dest_file)
    |> case do
      {:error, :enotsup} ->
        File.cp(src_file, dest_file, on_conflict: fn _, _ -> true end)
      {:error, :enoent} ->
        :ok
      :ok ->
        :ok
    end
  end

  defp clear_ref(model), do: Map.put(model, :ref, nil)

  # defp clear_ref(%{ref: nil} = model), do: model
  #
  # defp clear_ref(%{ref: ref} = model) do
  #   NIFAPI.call(:booster_free, ref)
  #   # => Segmentation fault
  #   Map.put(model, :ref, nil)
  # end
end
