defmodule LgbmEx.DataFrame do
  @moduledoc """
  DataFrame utils for internal use
  """

  alias Explorer.DataFrame

  @doc """
  Writes a dataframe to a csv for train by LightGBM.
  """
  def dump_csv!(df, {y_name, x_names}) do
    full_names = [y_name] ++ x_names

    df
    |> DataFrame.relocate(y_name, before: 0)
    |> DataFrame.select(full_names)
    |> DataFrame.dump_csv!(header: false)
    |> then(&{&1, y_name, x_names})
  end

  def dump_csv!(df, y_name) do
    x_names = DataFrame.names(df) -- [y_name]
    dump_csv!(df, {y_name, x_names})
  end

  def to_x(df, x_names) do
    [x_one | _] =
      x_list =
      df
      |> DataFrame.select(x_names)
      |> Nx.stack(axis: -1)
      |> Nx.to_list()

    # `Nx.stack` rejects string columns (silently), so alert if size is changed.
    num_columns_before = Enum.count(x_names)
    num_columns_after = Enum.count(x_one)

    if num_columns_before != num_columns_after do
      raise "maybe your prediction targets have string-type column"
    end

    x_list
  end
end
