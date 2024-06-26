defmodule LgbmEx.MixProject do
  use Mix.Project

  @version "0.0.2"

  def project do
    [
      app: :lgbm_ex,
      version: @version,
      elixir: "~> 1.15",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      compilers: [:elixir_make] ++ Mix.compilers(),
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Specifies which paths to compile per environment.
  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:elixir_make, "~> 0.4", runtime: false},
      {:jason, "~> 1.2"},
      # Data
      {:explorer, "~> 0.8.1"},
      {:scholar, "~> 0.2.1"},
      # Test
      {:mix_test_observer, "~> 0.1", only: [:dev, :test], runtime: false}
    ]
  end
end
