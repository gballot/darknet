# Define "split" and "ss" commands to switch to different user interface layouts
set max-value-size unlimited

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
