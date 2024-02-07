defmodule LgbmEx.ParameterFile do
  @moduledoc """
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
