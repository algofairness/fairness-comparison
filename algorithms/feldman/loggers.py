from time import strftime

def vprint(message, should_print):
  if should_print:
    print "{} ({})".format(message, strftime("%X"))
