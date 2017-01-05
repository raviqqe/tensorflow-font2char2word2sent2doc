require_relative './third/tensorflow-rakefile/tfrake.rb'
include TFRake


define_tasks('word2sent2doc',
             packages: %w(tensorflow-qnd),
             pytest_flags: %w(--ignore word2sent2doc/extenteten))
