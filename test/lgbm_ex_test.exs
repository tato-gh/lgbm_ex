defmodule LgbmExTest do
  use ExUnit.Case
  doctest LgbmEx

  alias LgbmEx.SampleDataIris

  describe "fit" do
    @describetag :tmp_dir

    test "returns model, num of iterations and evaluation value", %{
      tmp_dir: tmp_dir
    } do
      {x, y} = SampleDataIris.train_set()
      parameters = SampleDataIris.parameters()

      model = LgbmEx.new_model(tmp_dir)
      {model, num_iterations, learning_steps} = LgbmEx.fit(model, x, y, parameters)

      assert model
      assert num_iterations == 10
      # cannot get values because of unuse early_stopping
      assert Enum.count(learning_steps) == 1
    end
  end
end

