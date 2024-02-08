defmodule LgbmEx.Prediction do
  @moduledoc """
  TODO
  """

  alias LgbmEx.Interface

  @doc """
  Predict
  """
  def predict(%{ref: nil} = model, x) do
    {:ok, ref} = create_reference(model)
    predict(Map.put(model, :ref, ref), x)
  end

  def predict(_model, []), do: []

  def predict(model, [v | _] = features) when not is_list(v) do
    # one target
    json_arg = encode_to_json_charlist(%{row: features, ncol: Enum.count(features)})

    Interface.booster_predict_for_mat_single_row(model.ref, json_arg)
    |> decode_json_charlist()
    |> Map.get("result")
  end

  defp create_reference(%{files: %{model: file_path}}) do
    args = encode_to_json_charlist(%{file: file_path})
    Interface.booster_create_from_model_file(args)
  end

  defp encode_to_json_charlist(attrs) do
    Jason.encode!(attrs)
    |> String.to_charlist()
  end

  defp decode_json_charlist(charlist) do
    charlist
    |> List.to_string()
    |> Jason.decode!()
  end
end
