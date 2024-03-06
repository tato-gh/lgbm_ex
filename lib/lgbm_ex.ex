defmodule LgbmEx do
  @moduledoc """
  LgbmEx is a wrapper library for microsoft/LightGBM.

  - `fit` uses cli command
  - Others like `predict` use C API by NIF.
  """

  alias LgbmEx.Model
  alias LgbmEx.ModelFile
  alias LgbmEx.LightGBM
  alias LgbmEx.Splitter

  @doc """
  Returns new model struct work in workdir/cache.
  """
  def new_model(workdir) do
    Model.new_model(workdir)
  end

  @doc """
  Fit model.
  """
  def fit(model, {x_train, x_val}, {y_train, y_val}, parameters) do
    model = Model.setup_model(model, parameters, validation: true)
    ModelFile.write_data(model.files.train, x_train, y_train)
    ModelFile.write_data(model.files.validation, x_val, y_val)
    ModelFile.write_parameters(model.files.parameter, model.parameters)

    LightGBM.train(model)
    |> Model.complement_model_attrs()
  end

  def fit(model, x, y, parameters) do
    # Set validation data for output log on train phase.
    # Remove that after train because of duplication.
    model =
      fit(model, {x, x}, {y, y}, parameters)
      |> Map.update!(:parameters, & Keyword.put(&1, :valid_data, nil))

    File.rm!(model.files.validation)
    ModelFile.write_parameters(model.files.parameter, model.parameters)

    model
  end

  @doc """
  Fit model without eval (faster than `fit`).
  """
  def fit_without_eval(model, x, y, parameters) do
    model = Model.setup_model(model, parameters, validation: false)
    ModelFile.write_data(model.files.train, x, y)
    ModelFile.write_parameters(model.files.parameter, model.parameters)

    LightGBM.train(model)
    |> Model.complement_model_attrs()
  end

  @doc """
  Fit to existing data with given parameters.
  """
  def refit(model, parameters) do
    parameters = Keyword.merge(model.parameters, parameters)
    model = Model.setup_model(model, parameters)
    ModelFile.write_parameters(model.files.parameter, model.parameters)

    LightGBM.train(model)
    |> Model.complement_model_attrs()
  end

  @doc """
  Generate many models with given grid (parameters list).

  - Returns models generated at subdirs.
  - Data(train/test) are shared by hard link.
  - `grid` is like below.

  ```
  grid = [
    num_iterations: [5, 10],
    min_data_in_leaf: [2, 3]
  ]
  ```
  """
  def grid_search(model, grid) do
    combinations(grid)
    |> Enum.with_index(1)
    |> Enum.map(fn {parameters, index} ->
      submodel = Model.copy_model_as_sub(model, "grid-#{index}")
      refit(submodel, parameters)
    end)
  end

  defp combinations([]), do: [[]]

  defp combinations([{name, values} | rest]) do
    for sub <- combinations(rest), value <- values do
      [{name, value} | sub]
    end
  end

  @doc """
  Returns evaluation values each k-folding model.

  NOTE: Concat model train and validation to sample all data.
  """
  def cross_validate(model, x_test, k, folding_rule \\ :raw) do
    ModelFile.read_data(model.files.train)
    |> Kernel.++(ModelFile.read_data(model.files.validation) || [])
    |> Splitter.split(k, folding_rule)
    |> Enum.map(fn {train, val} ->
      {x_train, y_train} = train
      {x_val, y_val} = val

      model_cv =
        Model.copy_model(model, "cache_cross_validation")
        |> fit({x_train, x_val}, {y_train, y_val}, model.parameters)

      List.last(model_cv.learning_steps)
      |> case do
        {num_iterations, metric_val} ->
          %{
            train_size: Enum.count(y_train),
            val_size: Enum.count(y_val),
            num_iterations: num_iterations,
            last_value: metric_val,
            prediction: predict(model_cv, x_test),
            feature_importance_split: model_cv.feature_importance_split,
            feature_importance_gain: model_cv.feature_importance_gain
          }

        _ -> nil
      end
    end)
    |> Enum.filter(& &1)
  end

  @doc """
  Returns aggregated result of cross_validation.
  """
  def aggregate_cross_validation_results(results) do
    keys = ~w(num_iterations last_value prediction feature_importance_split feature_importance_gain)a

    results
    |> Enum.map(fn result -> Enum.map(keys, & Map.get(result, &1)) end)
    |> Enum.zip_reduce([], & &2 ++ [&1])
    |> Enum.zip(keys)
    |> Enum.map(fn
      {values, key} when key in [:num_iterations, :last_value] ->
        {key, calc_mean(values)}

      {values, key} when key in [:feature_importance_split, :feature_importance_gain] ->
        result =
          # k回分を統合してそれぞれ返す
          values
          |> Enum.zip_reduce([], & &2 ++ [&1])
          |> Enum.map(& calc_mean/1)

        {key, result}

      {values, :prediction} ->
        prediction =
          values
          |> Enum.zip_reduce([], & &2 ++ [&1])
          |> Enum.map(fn row_probs_list ->
            Enum.zip_reduce(row_probs_list, [], & &2 ++ [&1])
            |> Enum.map(& calc_mean/1)
          end)

        {:prediction, prediction}

      _ ->
        nil
    end)
    |> Enum.filter(& &1)
    |> Map.new()
  end

  @doc """
  Predict value by model.
  """
  def predict(model, x) do
    LightGBM.predict(model, x)
  end

  @doc """
  Save model.
  """
  def save_as(model, name) do
    Model.copy_model(model, name)
  end

  @doc """
  Load model from given workdir and name
  """
  def load_model(workdir, name) do
    model = Model.load_model(workdir, name)
    parameters = ModelFile.read_parameters(model.files.parameter)

    Map.put(model, :parameters, parameters)
    |> Model.complement_model_attrs()
  end

  @doc """
  Returns zip file path to export model.
  """
  def dump_zip(model) do
    dir = Path.join(model.workdir, model.name)
    model_files = Enum.map(model.files, & Path.basename(elem(&1, 1))) |> MapSet.new()
    existing_files = File.ls!(dir) |> MapSet.new()

    files = MapSet.intersection(model_files, existing_files)
            |> MapSet.to_list()
            |> Enum.map(& String.to_charlist/1)

    tmp_dir = System.tmp_dir!()
    zip_name = Path.join(tmp_dir, model.name <> ".zip") |> String.to_charlist()

    {:ok, zip_file_path} = :zip.create(zip_name, files, cwd: dir)
    List.to_string(zip_file_path)
  end

  @doc """
  Build model from zip.
  """
  def from_zip(zip_path, workdir, name) do
    zip_path = String.to_charlist(zip_path)
    dir = Path.join(workdir, name) |> String.to_charlist()
    :zip.extract(zip_path, cwd: dir)
    load_model(workdir, name)
  end

  defp calc_mean([]), do: nil

  defp calc_mean(values) do
    size = Enum.count(values)
    sum = Enum.sum(values)

    sum / size
  end
end
