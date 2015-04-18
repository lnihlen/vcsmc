`ifndef BUILD_FRAME_V
`define BUILD_FRAME_V

`include "tia_no_audio.v"

module build_frame();
  reg[1024*7:0] infile;
  reg[1024*7:0] outfile;
  reg[7:0] rom[0:16383];
  integer fdin, fdout, rom_size, address;
  reg clock;
  reg[3:0] state;
  parameter[3:0]
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
  reg phi2;
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

  initial begin
    address = 0;
    clock = 0;
    state = LOAD;
    d = 0;
    a = 0;
    phi2 = 0;

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
    if (clock) $fwrite(fdout, "%u", out_color);
  end

  always @(posedge phi_theta) begin
    if (state == LOAD) begin
      op = rom[address];
      address = address + 1;
      if (op == LDA || op == LDX || op == LDY || op == NOP) begin
        state = EXECUTE;
      end else if (op == JMP || op == STA || op == STX || op == STY) begin
        state = WAIT;
      end
    end else if (state == WAIT) begin
        state = EXECUTE;
    end else if (state == EXECUTE) begin
      case (op)
        8'h4c: begin  // jmp
          address = address + 1023;
          address = address & 32'hfffffc00;
        end
        8'ha9: begin  // lda
          reg_a = rom[address];
          address = address + 1;
        end
        8'ha2: begin  // ldx
          reg_x = rom[address];
          address = address + 1;
        end
        8'ha0: begin  // ldy
          reg_y = rom[address];
          address = address + 1;
        end
        8'h85: begin  // sta
          a = rom[address];
          d = reg_a;
          address = address + 1;
        end
        8'h86: begin  // stx
          a = rom[address];
          d = reg_x;
          address = address + 1;
        end
        8'h84: begin  // sty
          a = rom[address];
          d = reg_y;
          address = address + 1;
        end
        8'hea: begin  // nop
        end
        default: begin
          $display("unknown opcode %x at address %d", rom[address], address);
          $finish;
        end
      endcase
      state = LOAD;
    end

    if (address >= rom_size) begin
      $fclose(fdout);
      $finish;
    end
  end
endmodule  // build_frame

`endif  // BUILD_FRAME_V
