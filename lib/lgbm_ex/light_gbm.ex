defmodule LgbmEx.LightGBM do
  @moduledoc """
  TODO
  """

  alias LgbmEx.NIFAPI

  @doc """
  Train by cmd.
  """
  def train(model) do
    {_train_log, 0} = System.shell(get_cmd() <> " config=#{model.files.parameter}" <> " > #{model.files.train_log}")
    {:ok, ref} = NIFAPI.create_reference(model)
    Map.put(model, :ref, ref)
  end

  @doc """
  Predict by nif.
  """
  def predict(_model, []), do: []

  def predict(model, [v | _] = features) when not is_list(v) do
    # one target
    NIFAPI.call(:booster_predict_for_mat_single_row, model.ref, %{
      row: features
    })
  end

  def predict(model, x) do
    # multi target
    NIFAPI.call(:booster_predict_for_mat, model.ref, %{
      row: List.flatten(x),
      nrow: Enum.count(x)
    })
  end

  defp get_cmd do
    Path.join([:code.priv_dir(:lgbm_ex), "bin", "lightgbm"])
  end
end
