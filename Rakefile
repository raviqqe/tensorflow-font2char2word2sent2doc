require 'rake'
require_relative './third/tensorflow-rakefile/tfrake.rb'
include TFRake

define_tasks('font2char2word2sent2doc', define_pytest: false)

task_in_venv :pytest do
  sh 'wget http://dforest.watch.impress.co.jp/library/i/ipafont/10746/ipag00303.zip'
  sh 'unzip *.zip'
  sh 'cp ipag00303/*.ttf data/font.ttf'

  Dir.glob('font2char2word2sent2doc/**/*_test.py').each do |file|
    vsh :pytest, file
  end
end

task :examples

%w[word2sent2doc char2word2sent2doc font2char2word2sent2doc].each do |dir|
  name = "#{dir}_example"

  task_in_venv name do
    vsh "cd #{File.join 'examples', dir} && rake"
  end

  Rake::Task[:examples].enhance [name]
end

task test: %i[pytest examples]

task clobber: :clean do
  sh 'cd examples/var/dataset && rake clobber'
end
