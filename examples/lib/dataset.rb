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


file CHAR_FILE => WORD_FILE do |t|
  null_char = "\u25a1"
  unknown_char = "\ufffd"

  sh "echo #{null_char} > #{t.name}"
  sh "echo #{unknown_char} >> #{t.name}"

  sh %W(cat #{t.source} | sed 's/./\\0\\n/g' |
        grep -v -e #{null_char} -e #{unknown_char} -e '^[[:blank:]]*$' |
        sort -u >> #{t.name}).join(' ')
end
