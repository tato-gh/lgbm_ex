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
    |> then(& {&1, y_name, x_names})
  end

  def dump_csv!(df, y_name) do
    x_names = DataFrame.names(df) -- [y_name]
    dump_csv!(df, {y_name, x_names})
  end

  def to_x(df, x_names) do
    df
    |> DataFrame.select(x_names)
    |> Nx.stack(axis: -1)
    |> Nx.to_list()
  end
end
