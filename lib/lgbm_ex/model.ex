defmodule LgbmEx.Model do
  @moduledoc """
  """

  defstruct [:workdir, :name, :files, :parameters, :ref]

  @first_name "cache"

  @doc """
  Model with cache directory
  """
  def new_model(workdir) do
    %__MODULE__{workdir: workdir, name: @first_name}
    |> put_files()
    |> put_parameters()
  end

  @doc """
  Model with given directory
  """
  def load_model(workdir, name) do
    %__MODULE__{workdir: workdir, name: name}
    |> put_files()
  end

  @doc """
  copy model with given name directory
  """
  def copy_model(model, name) do
    dest_dir = Path.join(model.workdir, name)

    File.mkdir_p(dest_dir)
    Enum.each(model.files, fn {_key, src_file} ->
      dest_file = Path.join(dest_dir, Path.basename(src_file))
      File.cp(src_file, dest_file, on_conflict: fn _, _ -> true end)
    end)

    model
    |> Map.put(:name, name)
    |> put_files()
  end

  @doc """
  TODO
  """
  def merge_parameters(model, parameters) do
    custom_parameters =
      Keyword.drop(parameters, [:task, :data, :output_model, :label_column])

    Map.update!(model, :parameters, & Keyword.merge(&1, custom_parameters))
  end

  @doc """
  TODO
  """
  def with_validation(model) do
    Map.update!(model, :parameters, & Keyword.merge(&1, [
      valid_data: model.files.validation
    ]))
  end

  defp put_files(model) do
    dir = Path.join(model.workdir, model.name)

    if model.name == @first_name do
      File.rm_rf(dir)
      File.mkdir_p(dir)
    end

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
