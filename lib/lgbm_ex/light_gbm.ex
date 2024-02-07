defmodule LgbmEx.LightGBM do
  @moduledoc """
  TODO
  """

  @doc """
  TODO run by cmd
  """
  def train(model) do
    {output_log, 0} = System.shell(get_cmd() <> " config=#{model.files.parameter}")
    parse_output_log(output_log, Keyword.get(model.parameters, :metric))
  end

  defp parse_output_log(log, metric) do
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

  defp get_cmd do
    Path.join([:code.priv_dir(:lgbm_ex), "bin", "lightgbm"])
  end
end
