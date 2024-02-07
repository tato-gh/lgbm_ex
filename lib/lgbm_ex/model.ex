defmodule LgbmEx.Model do
  @moduledoc """
  """

  defstruct [:workdir, :name, :files, :parameters]

  @doc """
  model with cache directory
  """
  def cache_model(workdir) do
    name = "cache"

    %__MODULE__{workdir: workdir, name: name}
    |> put_files()
    |> put_parameters()
  end

  @doc """
  TODO
  """
  def merge_parameters(model, parameters) do
    custom_parameters =
      Keyword.drop(parameters, [:task, :data, :output_model, :label_column])

    Map.update!(model, :parameters, & Keyword.merge(&1, custom_parameters))
  end

  defp put_files(model) do
    dir = Path.join(model.workdir, model.name)

    File.rm_rf(dir)
    File.mkdir_p(dir)

    Map.put(model, :files, %{
      model: Path.join(dir, "model.txt"),
      train: Path.join(dir, "train.csv"),
      validation: Path.join(dir, "validation.csv"),
      parameter: Path.join(dir, "paramter.txt")
    })
  end

  defp put_parameters(model) do
    Map.put(model, :parameters, [
      task: "train",
      data: model.files.train,
      output_model: model.files.model,
      label_column: 0,
      saved_feature_importance_type: 1
    ])
  end
end
