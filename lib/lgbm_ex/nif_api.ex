defmodule LgbmEx.NIFAPI do
  @moduledoc """
  TODO
  """

  alias LgbmEx.NIF

  def call(action, ref, attrs) do
    args = encode_to_json_charlist(attrs)

    apply(NIF, action, [ref, args])
    |> decode_json_charlist()
    |> Map.get("result")
  end

  def create_reference(%{files: %{model: file_path}}) do
    args = encode_to_json_charlist(%{file: file_path})
    NIF.booster_create_from_model_file(args)
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
