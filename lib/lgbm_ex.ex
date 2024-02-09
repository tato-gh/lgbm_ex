defmodule LgbmEx do
  @moduledoc """
  LgbmEx is a wrapper library for microsoft/LightGBM.
  """

  alias LgbmEx.Model
  alias LgbmEx.Train
  alias LgbmEx.Parameter
  alias LgbmEx.Prediction
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
    Train.write_data(model.files.train, x_train, y_train)
    Train.write_data(model.files.validation, x_val, y_val)
    Parameter.write_data(model.files.parameter, model.parameters)

    {num_iterations, learning_steps} = LightGBM.train(model)
    {model, num_iterations, learning_steps}
  end

  def fit(model, x, y, parameters) do
    model = Model.setup_model(model, parameters, validation: false)
    Train.write_data(model.files.train, x, y)
    Parameter.write_data(model.files.parameter, model.parameters)

    {num_iterations, learning_steps} = LightGBM.train(model)
    {model, num_iterations, learning_steps}
  end

  @doc """
  Fit to existing data with given parameters
  """
  def refit(model, parameters) do
    parameters = Keyword.merge(model.parameters, parameters)
    model = Model.setup_model(model, parameters)
    Parameter.write_data(model.files.parameter, model.parameters)

    {num_iterations, learning_steps} = LightGBM.train(model)
    {model, num_iterations, learning_steps}
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
    parameters = Parameter.read_data(model.files.parameter)
    Map.put(model, :parameters, parameters)
  end

  @doc """
  Predict value by model.
  """
  def predict(model, x) do
    Prediction.predict(model, x)
  end
end
