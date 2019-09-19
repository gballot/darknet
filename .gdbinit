# Define "split" and "ss" commands to switch to different user interface layouts
define split
  layout split
  layout asm
  layout regs
  focus cmd
end

define ss
  layout split
  layout src
  layout regs
  focus cmd
end
