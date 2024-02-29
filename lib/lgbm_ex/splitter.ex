defmodule LgbmEx.Splitter do
  @moduledoc """
  Data splitter for cross_validation.
  """

  def split(list, k, :raw) do
    _split(list, k)
  end

  def split(list, k, :shuffle) do
    _split(Enum.shuffle(list), k)
  end

  def split(list, k, :sort) do
    # sort to pick variable values for each k
    box = Enum.map(1..k, fn _ -> [] end)

    Enum.sort_by(list, fn [y_value | _] -> y_value end)
    |> Enum.with_index()
    |> Enum.reduce(box, fn {row, index}, acc ->
      box_index = rem(index, k)
      List.update_at(acc, box_index, & [row] ++ &1)
    end)
    |> Enum.flat_map(& &1)
    |> _split(k)
  end

  defp _split(list, k) do
    chunk_size = round(Enum.count(list) / k)

    list
    |> Enum.chunk_every(chunk_size)
    |> Enum.map(& zip_x_y/1)
    |> fold()
  end

  defp zip_x_y(rows) do
    rows
    |> Enum.map(fn [label | features] -> [features, label] end)
    |> Enum.zip()
    |> Enum.map(& Tuple.to_list/1)
  end

  defp fold(list) do
    list
    |> Enum.with_index(0)
    |> Enum.map(fn {[x_val, y_val], index} ->
      validation_set = {x_val, y_val}

      train_set =
        List.delete_at(list, index)
        |> Enum.zip()
        |> Enum.map(& Tuple.to_list/1)
        |> then(fn [xs, ys] ->
          {Enum.flat_map(xs, & &1), List.flatten(ys)}
        end)

      {train_set, validation_set}
    end)
  end
end
