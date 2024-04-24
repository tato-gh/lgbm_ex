defmodule LgbmEx do
  @moduledoc """
  LgbmEx is a wrapper library for microsoft/LightGBM.

  - `fit` uses cli command
  - Others like `predict` use C API by NIF.
  """

  alias LgbmEx.Model

  def fit(model_name, {df_train, df_val}, feature_names, parameters) do
    {csv_train, y_name, x_names} = LgbmEx.DataFrame.dump_csv!(df_train, feature_names)
    {csv_val, _, _} = LgbmEx.DataFrame.dump_csv!(df_val, feature_names)
    parameters = merge_feature_names(parameters, y_name, x_names)

    Model.train(model_name, {csv_train, csv_val}, parameters)
  end

  def fit(model_name, df, feature_names, parameters) do
    {csv_train, y_name, x_names} = LgbmEx.DataFrame.dump_csv!(df, feature_names)
    parameters = merge_feature_names(parameters, y_name, x_names)

    Model.train(model_name, {csv_train}, parameters)
  end

  def fit_without_val(model_name, df, feature_names, parameters) do
    {csv_train, y_name, x_names} = LgbmEx.DataFrame.dump_csv!(df, feature_names)
    parameters = merge_feature_names(parameters, y_name, x_names)

    Model.train(model_name, csv_train, parameters)
  end

  def predict(model, df_or_list) do
    Model.predict(model, df_or_list)
  end

  def preproccessing_label_encode(df, name, mapping \\ nil) do
    mapping = _label_mapping(df[name], mapping)

    labels =
      Enum.reduce(mapping, df[name], fn {label, index}, acc ->
        Explorer.Series.replace(acc, label, to_string(index))
      end)
      |> Explorer.Series.cast(:category)

    {mapping, Explorer.DataFrame.put(df, name, labels)}
  end

  def gen_label_mapping(series), do: _label_mapping(series, nil)

  defp _label_mapping(series, nil) do
    series
    |> Explorer.Series.frequencies()
    |> Explorer.DataFrame.pull("values")
    |> Explorer.Series.to_list()
    |> Enum.with_index(0)
    |> Map.new()
  end

  defp _label_mapping(_series, mapping), do: mapping

  defp merge_feature_names(parameters, y_name, x_names) do
    Keyword.merge(parameters, y_name: y_name, x_names: x_names)
  end

  def load_model(name) do
    Model.load(name)
  end

  def refit_model(model, parameters) do
    Model.refit(model, parameters)
  end

  def copy_model(model, name) do
    Model.copy(model, name)
  end

  def zip_model(model) do
    {:ok, zip_file_path} = Model.zip(model)
    List.to_string(zip_file_path)
  end

  def unzip_model(path, name) do
    Model.unzip(path, name)
  end
end
