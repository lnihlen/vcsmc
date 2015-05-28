`ifndef BUILD_FRAME_V
`define BUILD_FRAME_V

`include "tia_no_audio.v"

// iverilog -Wall -o out/build_frame build_frame.v && vvp out/build_frame      \
// -lxt2 +input_file=/Users/luken/src/video/bin/frame-1.bin +output_file="foo.x"

module build_frame();
  reg[1024*7:0] infile;
  reg[1024*7:0] outfile;
  reg[7:0] rom[0:16383];
  integer fdin, fdout, rom_size, address, count, start_count;
  reg clock;
  wire phi2;
  reg[3:0] state;
  parameter[3:0]
      INIT = 4,
      WSYNC = 3,
      LOAD = 2,
      WAIT = 1,
      EXECUTE = 0;
  reg[7:0] reg_a;
  reg[7:0] reg_x;
  reg[7:0] reg_y;
  reg[7:0] op;

  parameter[7:0]
      JMP = 8'h4c,
      LDA = 8'ha9,
      LDX = 8'ha2,
      LDY = 8'ha0,
      NOP = 8'hea,
      STA = 8'h85,
      STX = 8'h86,
      STY = 8'h84;

  reg[7:0] d;
  reg[5:0] a;
  reg rw;
  wire blk_bar;
  wire[2:0] l;
  wire[3:0] c;
  wire syn;
  wire rdy;
  wire phi_theta;
  tia_no_audio no_audio(.d(d),
                        .a(a),
                        .osc(clock),
                        .phi2(phi2),
                        .rw(rw),
                        .blk_bar(blk_bar),
                        .l(l),
                        .c(c),
                        .syn(syn),
                        .rdy(rdy),
                        .phi_theta(phi_theta));

  wire[7:0] out_color;
  assign out_color[0] = 0;
  assign out_color[1] = l[0];
  assign out_color[2] = l[1];
  assign out_color[3] = l[2];
  assign out_color[4] = c[0];
  assign out_color[5] = c[1];
  assign out_color[6] = c[2];
  assign out_color[7] = c[3];

  assign #10 phi2 = clock;

  initial begin
    address = 0;
    clock = 1;
    count = -1;
    rw = 1;
    state = INIT;
    d = 0;
    a = 6'b111111;

    $dumpfile("out/build_frame.lxt");
    $dumpvars(0, build_frame);

    if (!$value$plusargs("input_file=%s", infile)) begin
      $display("no input file specified with +input_file=");
      $finish;
    end
    if (!$value$plusargs("output_file=%s", outfile)) begin
      $display("no output file specified with +output_file=");
      $finish;
    end
    fdin = $fopen(infile, "r");
    if (fdin == 0) begin
      $display("error opening input file %0s", infile);
      $finish;
    end
    rom_size = $fread(rom, fdin, 0, 16383);
    $fclose(fdin);
    fdout = $fopen(outfile, "w");
    if (fdout == 0) begin
      $display("error opening output file %0s", outfile);
      $finish;
    end
  end

  always #100 begin
    clock = ~clock;
    if (!clock) begin
      count = count + 1;
      #1
      $fwrite(fdout, "%02x", out_color);
    end
  end

  always @(posedge phi_theta) begin
    #10
    if (state == INIT) begin
      state = LOAD;
    end else if (state == WSYNC && (rdy === 0)) begin
      rw = 1;
    end else if (state == LOAD || (state == WSYNC && rdy === 1)) begin
      if (address >= rom_size) begin
        $fclose(fdout);
        $dumpflush;
        $finish;
      end
      op = rom[address];
      address = address + 1;
      start_count = count;
      rw = 1;
      if (op == LDA || op == LDX || op == LDY || op == NOP) begin
        state = EXECUTE;
      end else if (op == JMP || op == STA || op == STX || op == STY) begin
        state = WAIT;
      end
    end else if (state == WAIT) begin
        state = EXECUTE;
    end else if (state == EXECUTE) begin
      case (op)
        JMP: begin
//          $display("%d %x: jmp", start_count, address);
          address = address + 1023;
          address = address & 32'hfffffc00;
        end
        LDA: begin
//          $display("%d %x: lda %x", start_count, address, rom[address]);
          reg_a = rom[address];
        end
        LDX: begin
//          $display("%d %x: ldx %x", start_count, address, rom[address]);
          reg_x = rom[address];
        end
        LDY: begin
//          $display("%d %x: ldy %x", start_count, address, rom[address]);
          reg_y = rom[address];
        end
        STA: begin
//          $display("%d %x: sta %x", start_count, address, rom[address]);
          a = rom[address];
          d = reg_a;
        end
        STX: begin
//          $display("%d %x: stx %x", start_count, address, rom[address]);
          a = rom[address];
          d = reg_x;
        end
        STY: begin
//          $display("%d %x: sty %x", start_count, address, rom[address]);
          a = rom[address];
          d = reg_y;
        end
        NOP: begin
//          $display("%d %x: nop", start_count, address);
        end
        default: begin
//          $display("unknown opcode %x at address %d", rom[address], address);
          $finish;
        end
      endcase
      #1
      if (op == STA || op == STX || op == STY) begin
        rw = 0;
        if (a == 8'h02) state = WSYNC;
        else state = LOAD;
      end else state = LOAD;

      if (op != JMP && op != NOP) address = address + 1;
    end
  end

endmodule  // build_frame

`endif  // BUILD_FRAME_V
