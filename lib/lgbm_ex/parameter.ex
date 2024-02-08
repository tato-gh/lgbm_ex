defmodule LgbmEx.Parameter do
  @moduledoc """
  TODO
  """

  @doc """
  TODO
  """
  def write_data(file_path, parameters) do
    params_str =
      Enum.map(parameters, fn {key, value} -> "#{key} = #{value}" end)
      |> Enum.join("\n")

    File.write!(file_path, params_str <> "\n")
  end

  @doc """
  TODO
  """
  def read_data(file_path) do
    File.read!(file_path)
    |> String.split("\n")
    |> Enum.reduce([], fn row, acc ->
      String.split(row, "=", trim: true)
      |> Enum.map(& String.trim/1)
      |> case do
        [key, value] -> Keyword.put(acc, :"#{key}", conv_type(value))
        _ -> acc
      end
    end)
  end

  defp conv_type(value) do
    {Integer.parse(value), Float.parse(value)}
    |> case do
      {{v, ""}, _} -> v
      {_, {v, ""}} -> v
      _ -> value
    end
  end
end
