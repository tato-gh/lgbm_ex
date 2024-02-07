defmodule LgbmEx do
  @moduledoc """
  LgbmEx is a wrapper library for microsoft/LightGBM.
  """

  alias LgbmEx.Model
  alias LgbmEx.TrainFile
  alias LgbmEx.ParameterFile
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
    model =
      model
      |> Model.merge_parameters(parameters)
      |> Model.with_validation()

    TrainFile.write_data(model.files.train, x_train, y_train)
    TrainFile.write_data(model.files.validation, x_val, y_val)
    ParameterFile.write(model.files.parameter, model.parameters)

    {num_iterations, learning_steps} = LightGBM.train(model)
    {model, num_iterations, learning_steps}
  end

  def fit(model, x, y, parameters) do
    model = Model.merge_parameters(model, parameters)
    TrainFile.write_data(model.files.train, x, y)
    ParameterFile.write(model.files.parameter, model.parameters)

    {num_iterations, learning_steps} = LightGBM.train(model)
    {model, num_iterations, learning_steps}
  end

  @doc """
  Save model.
  """
  def save_as(model, name) do
    Model.copy_model(model, name)
  end
end
