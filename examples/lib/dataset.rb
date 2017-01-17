require_relative './config'


file VAR_DIR do
  ln_s SHARED_VAR_DIR, VAR_DIR
end


task :dataset => VAR_DIR do
  sh "cd #{DATASET_DIR} && rake"
end


file WORD_FILE => :dataset do |t|
  sh "echo '<null>' > #{t.name}"
  sh "echo '<unknown>' >> #{t.name}"
  sh "cat #{DATASET_DIR}/aclImdb/imdb.vocab >> #{t.name}"
end
