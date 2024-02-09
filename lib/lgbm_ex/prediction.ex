defmodule LgbmEx.Prediction do
  @moduledoc """
  TODO
  """

  alias LgbmEx.NIFAPI

  @doc """
  Predict
  """
  def predict(%{ref: nil} = model, x) do
    {:ok, ref} = NIFAPI.create_reference(model)
    predict(Map.put(model, :ref, ref), x)
  end

  def predict(_model, []), do: []

  def predict(model, [v | _] = features) when not is_list(v) do
    # one target
    NIFAPI.call(:booster_predict_for_mat_single_row, model.ref, %{
      row: features,
      ncol: Enum.count(features)
    })
  end

  def predict(model, x) do
    # multi target
    NIFAPI.call(:booster_predict_for_mat, model.ref, %{
      row: List.flatten(x),
      ncol: hd(x) |> Enum.count(),
      nrow: Enum.count(x)
    })
  end
end
