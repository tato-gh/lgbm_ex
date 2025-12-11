defmodule LgbmEx.NIFAPI do
  @moduledoc """
  Nif api interface.
  """

  alias LgbmEx.NIF

  def call(action, ref) do
    apply(NIF, action, [ref])
    |> decode_json_charlist()
    |> fetch_result!(action)
  end

  def call(action, ref, args) when is_list(args) do
    apply(NIF, action, [ref | args])
    |> decode_json_charlist()
    |> fetch_result!(action)
  end

  def call(action, ref, attrs) when is_map(attrs) do
    args = encode_to_json_charlist(attrs)

    apply(NIF, action, [ref, args])
    |> decode_json_charlist()
    |> fetch_result!(action)
  end

  def create_reference(%{files: %{model: file_path}}) do
    args = encode_to_json_charlist(%{file: file_path})

    case NIF.booster_create_from_model_file(args) do
      {:ok, ref} ->
        {:ok, ref}

      {:error, reason} ->
        raise RuntimeError, message: format_error(:booster_create_from_model_file, reason)
    end
  end

  defp encode_to_json_charlist(attrs) do
    Jason.encode!(attrs)
    |> String.to_charlist()
  end

  defp decode_json_charlist(charlist) when is_list(charlist),
    do: charlist |> to_string() |> Jason.decode!()

  defp decode_json_charlist(binary) when is_binary(binary),
    do: Jason.decode!(binary)

  defp fetch_result!(%{"error" => message}, action) do
    raise RuntimeError, message: format_error(action, message)
  end

  defp fetch_result!(%{"result" => result}, _action), do: result
  defp fetch_result!(response, _action), do: response

  defp format_error(action, reason) do
    reason =
      case reason do
        list when is_list(list) -> to_string(list)
        binary when is_binary(binary) -> binary
        other -> inspect(other)
      end

    "NIF #{action} failed: #{reason}"
  end
end
