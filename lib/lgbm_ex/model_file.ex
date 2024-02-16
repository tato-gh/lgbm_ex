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
  def read_data(file_path) do
    File.exists?(file_path)
    |> if do
      File.read!(file_path)
      |> String.split("\n", trim: true)
      |> Enum.map(& String.split(&1, ","))
    else
      nil
    end
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

  @doc """
  TODO
  """
  def parse_train_log(file_path, metric) do
    {:ok, log} = File.read(file_path)
    _parse_train_log(log, metric)
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

  defp _parse_train_log(log, metric) do
    # heuristic logic, good luck!
    log
    |> String.split("\n")
    |> Enum.map_reduce(0, fn row, acc ->
      score = parse_score(row, metric)
      iteration = parse_iteration(row)

      case {score, iteration} do
        {nil, nil} ->
          # other log, skip row
          {nil, acc}

        {nil, iteration} ->
          # iteration set
          {nil, iteration}

        {score, nil} ->
          # score got on current acc(= iteration)
          {{acc, score}, nil}
      end
    end)
    |> then(fn {steps, num_iterations} ->
      {num_iterations, Enum.filter(steps, & &1 && elem(&1, 0))}
    end)
  end

  defp parse_score(row, metric) do
    Regex.scan(~r/#{metric} : ([-\.\d]+)/, row)
    |> case do
      [[_, matched]] -> String.to_float(matched)
      [] -> nil
    end
  end

  defp parse_iteration(row) do
    Regex.scan(~r/finished iteration (\d+)/, row)
    |> case do
      [[_, matched]] -> String.to_integer(matched)
      [] -> nil
    end
  end
end
