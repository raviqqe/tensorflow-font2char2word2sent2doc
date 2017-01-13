require_relative './third/tensorflow-rakefile/tfrake.rb'
include TFRake


define_tasks('font2char2word2sent2doc',
             define_pytest: false,
             packages: %i(listpad))


task_in_venv :pytest do
  Dir.glob('font2char2word2sent2doc/**/*_test.py').each do |file|
    vsh :pytest, file
  end
end


task_in_venv :examples do
  %w(word2sent2doc).each do |dir|
    vsh "cd #{File.join 'examples', dir} && rake"
  end
end


task :test => %i(pytest examples)
