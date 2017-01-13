require_relative './config'


file VAR_DIR do
  ln_s SHARED_VAR_DIR, VAR_DIR
end


task :dataset => VAR_DIR do
  sh "cd #{DATASET_DIR} && rake"
end
