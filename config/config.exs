import Config

config :lgbm_ex,
  workdir: System.get_env("LGBM_EX_WORKDIR") || Path.join(System.tmp_dir!(), "lgbm_ex")
