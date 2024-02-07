defmodule LgbmEx.ParameterFile do
  @moduledoc """
    model = Model.merge_parameters(model, parameters)
    TrainFile.write_data(model.files.train, x, y)
    ParameterFile.write(model.files.parameter, model.parameters)
    {num_iterations, learning_steps} = LightGBM.train(model)

    {model, num_iterations, learning_steps}
  TODO
  """

  @doc """
  TODO
  """
  def write(file_path, parameters) do
    params_str =
      Enum.map(parameters, fn {key, value} -> "#{key} = #{value}" end)
      |> Enum.join("\n")

    File.write!(file_path, params_str <> "\n")
  end
end
