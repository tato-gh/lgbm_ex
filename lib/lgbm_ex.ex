defmodule LgbmEx do
  @moduledoc """
  LgbmEx is a wrapper library for microsoft/LightGBM.
  """

  alias LgbmEx.Model
  alias LgbmEx.ModelFile
  alias LgbmEx.LightGBM

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
    model = Model.setup_model(model, parameters, validation: false)
    ModelFile.write_data(model.files.train, x, y)
    ModelFile.write_parameters(model.files.parameter, model.parameters)

    LightGBM.train(model)
    |> Model.complement_model_attrs()
  end

  @doc """
  Fit to existing data with given parameters
  """
  def refit(model, parameters) do
    parameters = Keyword.merge(model.parameters, parameters)
    model = Model.setup_model(model, parameters)
    ModelFile.write_parameters(model.files.parameter, model.parameters)

    LightGBM.train(model)
    |> Model.complement_model_attrs()
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
  Export model as zip.
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
end
