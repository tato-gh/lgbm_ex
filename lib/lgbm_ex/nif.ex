defmodule LgbmEx.NIF do
  @moduledoc """
  Nif interface.
  """

  @on_load {:load_nif, 0}
  @compile {:autoload, false}

  def load_nif do
    Application.app_dir(:lgbm_ex, "priv/liblgbmex")
    |> to_charlist()
    |> :erlang.load_nif(0)
  end

  @doc """
  Load model.
  Returns resource(model) reference handle.
  """
  def booster_create_from_model_file(_charlist_json), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Predict one element.
  Returns result as charlist.
  """
  def booster_predict_for_mat_single_row(_reference, _charlist_json), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Predict elements.
  Returns result as charlist.
  """
  def booster_predict_for_mat(_reference, _charlist_json), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Get booster num of classes.
  """
  def booster_get_num_classes(_reference), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Get booster num of features.
  """
  def booster_get_num_features(_reference), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Get booster current iterations.
  """
  def booster_get_current_iteration(_reference), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Get booster parameters
  Returns result as charlist.
  """
  def booster_get_loaded_param(_reference), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Get booster feature importance
  Returns result as charlist.
  """
  def booster_feature_importance(_reference), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Free booster.
  """
  def booster_free(_reference), do: :erlang.nif_error(:nif_not_loaded)
end
