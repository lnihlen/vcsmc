// The z26 sources keep a lot of state in global scope. z26/z26.c includes the
// other C files directly. We mirror that pattern and modify the included c
// files to refer to a provided z26_state point.

#include "libz26/libz26.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// taken from z26.c for broad code compatibility
typedef unsigned int       dd;  /* define double */
typedef unsigned short int dw;  /* define word */
typedef unsigned char      db;  /* define byte */

struct z26_state {
  // Below moved from z26/c_tialine.c
  dd TIACollide;
  dd TIA_Last_Pixel;
  dd TIA_Display_HBlank;
  dd TIA_HMOVE_Setup;
  dd TIA_HMOVE_Clock;
  db TIA_HMOVE_Latches;
  db TIA_HMOVE_DoMove;
  db TIA_HMP0_Value;
  db TIA_HMP1_Value;
  db TIA_HMM0_Value;
  db TIA_HMM1_Value;
  db TIA_HMBL_Value;
  db TIA_Pixel_State;
  db TIA_Mask_Objects;
  db TIA_Do_Output;
  db CTRLPF_PF_Reflect;
  db Current_PF_Pixel;
  db NUSIZ_P0_width;
  db NUSIZ_P1_width;
  db NUSIZ_M0_width;
  db NUSIZ_M1_width;
  db CTRLPF_BL_width;
  db TIA_RESMP0;
  db TIA_RESMP1;
  db TIA_VDELP0;
  db TIA_VDELP1;
  db TIA_VDELBL;
  dd TIA_REFP0;
  dd TIA_REFP1;
  dd TIA_GRP0_new;
  dd TIA_GRP0_old;
  dd TIA_GRP1_new;
  dd TIA_GRP1_old;
  dd TIA_ENAM0;
  dd TIA_ENAM1;
  dd TIA_ENABL_new;
  dd TIA_ENABL_old;
  db NUSIZ0_number;
  db NUSIZ1_number;
  db CTRLPF_Score;
  db CTRLPF_Priority;
  dd TIA_P0_counter;
  dd TIA_P1_counter;
  dd TIA_M0_counter;
  dd TIA_M1_counter;
  dd TIA_BL_counter;
  db TIA_Delayed_Write;
  dw TIA_Colour_Table[4];
  dd TIA_P0_counter_reset;
  dd TIA_P1_counter_reset;
  dd TIA_M0_counter_reset;
  dd TIA_M1_counter_reset;
  dd TIA_BL_counter_reset;
  dd TIA_Playfield_Value;
  dd TIA_REFPF_Flag;
  db TIA_VBLANK;
  dw LoopCount;
  dw CountLoop;
  dd Pointer_Index_P0;
  dd Pointer_Index_P1;
  dd Pointer_Index_M0;
  dd Pointer_Index_M1;
  dd Pointer_Index_BL;
  db* TIA_P0_Line_Pointer;
  db* TIA_P1_Line_Pointer;
  db* TIA_M0_Line_Pointer;
  db* TIA_M1_Line_Pointer;
  db* TIA_BL_Line_Pointer;
  dd TIA_Playfield_Bits;

  // Below copied from z26/globals.c, which we do not include.
  db* ScreenBuffer;
  dw *DisplayPointer;
  int ScanLine;
  dw reg_pc;
  db reg_sp;
  db reg_a;
  db flag_C;
  db reg_x;
  db reg_y;
  db flag_Z;
  db flag_N;
  db flag_D;
  db flag_V;
  db flag_I;
  db flag_B;
  db RCycles;
  db RClock;
  db TriggerWSYNC;
  db DataBus;
  dw AddressBus;
  db RiotRam[128];
  dd ChargeCounter;
  int VSyncFlag;
  dd LinesInFrame;
  dd PrevLinesInFrame;
  int Frame;
  int VBlank;

  // Below moved from z26/cpu.m4.
  db dummy_flag;
  db dummy_high;
  db dummy_low;

  // Below moved from z26/c_riot.c
  dd Timer;
  void (* TimerReadVec)(struct z26_state* s);
  db DDR_A;
  db DDR_B;

  // ROM data used for simulation.
  const uint8_t* ROM_data;
  size_t ROM_size;
};

// constants defined in z26/globals.c, which we do not include.
const dd MaxLines = 256;
#define CYCLESPERSCANLINE 76
// Supposed to be seeded, here set to a constant Unix Epoch to avoid
// randomization.
const int Seconds = 1448577872;
const int TraceCount = 0;
const dd ChargeTrigger0[4] = { 240, 240, 240, 240 };
// constants defined in z26/position.c, which we do not include.
const dd TopLine = 18;
const dd BottomLine = TopLine + MaxLines;
const db InputLatch[2] = {0, 0};

// Read and write function pointer tables, some cart architecture specific
// pointers. Not per simulation run but perhaps per cart architecture. Currently
// initialized by init_z26_global_tables() (by calling c_init.c::InitData())
void (* ReadAccess[0x10000])(struct z26_state* s);
void (* WriteAccess[0x10000])(struct z26_state* s);
void (* TIARIOTReadAccess[0x1000])(struct z26_state* s);
void (* TIARIOTWriteAccess[0x1000])(struct z26_state* s);

void ReadROM4K(struct z26_state* s) {
  // ** VCSMC bank-switching strategy goes here.
  dw address = s->AddressBus & 0xfff;
  if (address >= s->ROM_size) {
    printf("warning: invalid ROM read at address %x\n", address);
    s->DataBus = 0;
    return;
  }
  // For the moment just do the unimaginative thing of reading the requested
  // ROM address, no support for ROMs greater than 4K.
  s->DataBus = s->ROM_data[address];
}

void WriteROM4K(struct z26_state* s) {
  // VCSMC ROMS are read-only.
  dw address = s->AddressBus & 0xfff;
  printf("warning: ROM write attempt detected at address %x\n", address);
}

// TODO: bring in sound simulation later.
void QueueSoundBytes(struct z26_state* s) {
}

void H_AUDC0(struct z26_state* s) {
}


void H_AUDC1(struct z26_state* s) {
}


void H_AUDF0(struct z26_state* s) {
}


void H_AUDF1(struct z26_state* s) {
}


void H_AUDV0(struct z26_state* s) {
}


void H_AUDV1(struct z26_state* s) {
}


// From z26/cpujam.c, made into a no-op.
void jam(struct z26_state* s) {
}

// From z26/c_trace.c, made into a no-op.
void TraceInstruction() {
}

// From z26/controls.c, no-op.
void TestLightgunHit(dd RClock, dd ScanLine) {
}

#include "z26/c_riot.c"
#include "m4/cpu.c"
#include "z26/c_tialine.c"
#include "z26/c_tiawrite.c"
#include "z26/c_init.c"

void init_z26_global_tables(void) {
  // Function from c_init.c that initializes the read and write function pointer
  // tables, as well as calling other Init() functions, some of which are
  // disabled.
  InitData();
}

struct z26_state* init_z26_state(uint8_t* output_picture) {
  struct z26_state* s = malloc(sizeof(struct z26_state));
  memset(s, 0, sizeof(struct z26_state));
  // Only non-zero setting copied here.
  s->TIA_HMOVE_Clock = 16;
  s->TIA_HMP0_Value = 8;
  s->TIA_HMP1_Value = 8;
  s->TIA_HMM0_Value = 8;
  s->TIA_HMM1_Value = 8;
  s->TIA_HMBL_Value = 8;
  s->TIA_Do_Output = 1;

  s->TIA_P0_Line_Pointer = TIA_P0_Table[0];
  s->TIA_P1_Line_Pointer = TIA_P1_Table[0];
  s->TIA_M0_Line_Pointer = TIA_M0_Table[0];
  s->TIA_M1_Line_Pointer = TIA_M1_Table[0];
  s->TIA_BL_Line_Pointer = TIA_BL_Table[0];
  s->TIA_Mask_Objects = 0x3f;

  s->ScreenBuffer = output_picture;
  s->DisplayPointer = (dw*)output_picture;

  s->flag_B = 0x10;

  // Initialization copied from c_riot.c::Init_Riot()
  s->Timer = START_TIME;
  s->TimerReadVec = &ReadTimer1024;

  s->LinesInFrame = kLibZ26ImageHeight;
  s->ScanLine = 1;

  return s;
}

// Loosely following the structure in z26/2600core.c::ScanFrame()
void simulate_single_frame(const uint8_t* byte_code,
                           size_t size,
                           uint8_t* output_picture) {
  struct z26_state* s = init_z26_state(output_picture);
  s->ROM_data = byte_code;
  s->ROM_size = size;

  // Simulate frame.
  while (s->Frame == 0) {
    nTIALineTo(s);
    ++s->ScanLine;
    s->RClock -= CYCLESPERSCANLINE;
  }

  // Clean up.
  free(s);
}
