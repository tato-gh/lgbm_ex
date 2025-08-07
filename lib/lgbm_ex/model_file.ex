defmodule LgbmEx.ModelFile do
  @moduledoc """
  ModelFile for LightGBM CLI outputs.
  """

  def write!(path, binary) do
    File.write!(path, binary)
  end

  def write_parameters!(path, parameters) do
    params_str =
      Enum.map(parameters, fn
        {key, values} when is_list(values) ->
          "#{key} = #{Enum.join(values, ",")}"

        {key, value} ->
          "#{key} = #{value}"
      end)
      |> Enum.join("\n")

    write!(path, params_str <> "\n")
  end

  def read_parameters!(path) do
    File.read!(path)
    |> String.split("\n")
    |> Enum.reduce([], fn row, acc ->
      String.split(row, "=", trim: true)
      |> Enum.map(&String.trim/1)
      |> case do
        ["x_names", value] -> Keyword.put(acc, :x_names, String.split(value, ","))
        [key, value] -> Keyword.put(acc, :"#{key}", conv_type(value))
        _ -> acc
      end
    end)
  end

  def maybe_make_hard_link(src_file, dest_file) do
    File.ln(src_file, dest_file)
    |> case do
      {:error, :enotsup} ->
        File.cp(src_file, dest_file, on_conflict: fn _, _ -> true end)

      {:error, :eexist} ->
        :ok

      {:error, :enoent} ->
        :ok

      :ok ->
        :ok
    end
  end

  @doc """
  Returns num_iterations and learning_steps.
  """
  def parse_train_log(path, metric) do
    {:ok, log} = File.read(path)
    _parse_train_log(log, metric)
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
      {num_iterations, Enum.filter(steps, &(&1 && elem(&1, 0)))}
    end)
  end

  defp parse_score(row, metric) do
    Regex.scan(~r/#{metric} : ([-\.\d]+)/, row)
    |> case do
      [[_, matched]] -> conv_type(matched)
      [] -> nil
    end
  end

  defp parse_iteration(row) do
    get_early_stopping_best_iteration(row) || get_finished_iteration(row)
  end

  defp get_early_stopping_best_iteration(row) do
    Regex.scan(~r/the best iteration round is (\d+)/, row)
    |> case do
      [[_, matched]] -> String.to_integer(matched)
      _ -> nil
    end
  end

  defp get_finished_iteration(row) do
    Regex.scan(~r/finished iteration (\d+)/, row)
    |> case do
      [[_, matched]] -> String.to_integer(matched)
      _ -> nil
    end
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
