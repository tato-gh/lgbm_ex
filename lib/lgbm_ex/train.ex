defmodule LgbmEx.Train do
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
end
