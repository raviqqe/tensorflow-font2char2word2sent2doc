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


file FONT_FILE => VAR_DIR do |t|
  sh "wget -O #{t.source}/font.zip http://dforest.watch.impress.co.jp/library/i/ipafont/10746/ipag00303.zip"
  sh "cd #{t.source} && unzip font.zip"
  sh "cp #{VAR_DIR}/ipag00303/ipag.ttf #{t.name}"
end
