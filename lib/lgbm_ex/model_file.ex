defmodule LgbmEx.ModelFile do
  @moduledoc """
  TODO
  """

  @doc """
  TODO
  """
  def write_data(file_path, x, y) do
    csv = join_data(x, y)
    File.write!(file_path, csv)
  end

  @doc """
  TODO
  """
  def write_parameters(file_path, parameters) do
    params_str =
      Enum.map(parameters, fn {key, value} -> "#{key} = #{value}" end)
      |> Enum.join("\n")

    File.write!(file_path, params_str <> "\n")
  end

  @doc """
  TODO
  """
  def read_parameters(file_path) do
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

  defp join_data(x, y) do
    Enum.zip(x, y)
    |> Enum.map(fn {features, label} -> "#{label}," <> join_values(features, "") end)
    |> Enum.join("\n")
    |> Kernel.<>("\n")
  end

  defp join_values([only_one], _acc) do
    "#{only_one ||"NA"}"
  end

  defp join_values([head, tail], acc) do
    acc <> "#{head || "NA"},#{tail || "NA"}"
  end

  defp join_values([head | tail], acc) do
    join_values(tail, acc <> "#{head || "NA"},")
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
