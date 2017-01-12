require_relative './config.rb'


task :clean do
  rm_rf OUTPUT_DIR
end


task :clobber do
  sh 'cd .. && git clean -dfx'
end
