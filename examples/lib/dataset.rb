require_relative './config'


file VAR_DIR do
  ln_s SHARED_VAR_DIR, VAR_DIR
end


directory "#{DATASET_DIR}/aclImdb.json" => VAR_DIR do
  sh "cd #{DATASET_DIR} && rake"
end
