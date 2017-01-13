require_relative './config.rb'


task :clean do
  rm_rf OUTPUT_DIR
end


task :clobber do
  sh "git clean -dfx"
  shared_dataset_dir = "../#{DATASET_DIR}"
  sh "cd #{shared_dataset_dir} && rake clobber" if File.directory? shared_dataset_dir
end
