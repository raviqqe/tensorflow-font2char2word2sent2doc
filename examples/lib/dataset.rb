require_relative './config.rb'

IMDB_TAR = 'aclImdb_v1.tar.gz'
DATASET_FILE = "#{SHARED_VAR_DIR}/#{IMDB_TAR}"
DATASET_DIR = "#{SHARED_VAR_DIR}/aclImdb"


file DATASET_FILE do |t|
  mkdir_p File.dirname(t.name)
  sh "wget -O #{t.name} http://ai.stanford.edu/~amaas/data/sentiment/#{IMDB_TAR}"
end


directory DATASET_DIR => DATASET_FILE do |t|
  sh "cd #{File.dirname(t.source)} && tar xf #{t.source}"
end


file VAR_DIR => DATASET_DIR do |t|
  ln_s SHARED_VAR_DIR,t.name
end
