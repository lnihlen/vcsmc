/*

	tiawrite.c -- handlers for writes to and reads from TIA registers


 z26 is Copyright 1997-2011 by John Saeger and contributors.  
 contributors.	 z26 is released subject to the terms and conditions of the 
 GNU General Public License Version 2 (GPL).  z26 comes with no warranty.
 Please see COPYING.TXT for details.
*/

/*
	TIA read handlers
*/

void ReadCollision(struct z26_state* s){

	CatchUpPixels(s);
	s->DataBus = (s->DataBus & 0x3f) |
	          ((s->TIACollide >> ((s->AddressBus & 0x7) << 1)) & 0x3) << 6;
}

void ReadAnalogINPT(struct z26_state* s){

	if(ChargeTrigger0[s->AddressBus & 0x3] <= s->ChargeCounter)
		s->DataBus = (s->DataBus & 0x3f) | 0x80;
}

void ReadDigitalINPT(struct z26_state* s){

	TestLightgunHit(s->RClock, s->ScanLine);
	s->DataBus = (s->DataBus & 0x3f) | InputLatch[s->AddressBus & 0x1];
}

void ReadTIAdummy(struct z26_state* s){

	s->DataBus &= 0x3f;	/* TIA only sets the two topmost bits */
}

/*
	TIA write handlers
*/

/* <LN>
void AdjustPalette() {

	if (Frame > 256) return;
	if (UserPaletteNumber != 0xff) return;
	if (GamePaletteNumber != 0xff) return;
	if (LinesInFrame != PrevLinesInFrame)
	{
		if (Frame != 5) return;		// force adjustment of unstable games (once)
	}

	if (LinesInFrame < 287)
	{	// NTSC
		if (PaletteNumber == 0) return;
		PaletteNumber = 0;
	}
	else
	{	// PAL
		if (PaletteNumber == 1) return;
		PaletteNumber = 1;
	}
	srv_SetPalette();
	ClearScreenBuffers();
	position_game();

}

*/

void H_VSYNC(struct z26_state* s){

	if(s->DataBus & 0x02){
	
		if (s->VSyncFlag) return;	// somebody hitting VSYNC more than necessary
		
		s->PrevLinesInFrame = s->LinesInFrame;
		s->LinesInFrame = s->ScanLine - 1;
		s->ScanLine = 1;
		if (s->LinesInFrame > 3) s->Frame++;	// ignore quick double-hit (pickpile)
		
// <LN>		AdjustPalette();
	}
	s->VSyncFlag = s->DataBus & 0x02;
}


void H_VBLANK(struct z26_state* s){
	
	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	s->VBlank = s->DataBus;
//	TIA_Mask_Objects = ((s->DataBus & 0x02) << 5) | (TIA_Mask_Objects & 0x3f);
	s->TIA_VBLANK = s->DataBus & 0x02;
	if(s->DataBus & 0x80) s->ChargeCounter = 0;
}


void H_WSYNC(struct z26_state* s){

 	// Don't halt the CPU, if we're already at the end of the line
	if(s->RClock != CYCLESPERSCANLINE) s->TriggerWSYNC = 1;
 	// WSYNC only halts the CPU on read cycles so we handle it in cpu.m4
}


void H_Null(struct z26_state* s){

}


void H_NUSIZ0(struct z26_state* s){
//	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	s->NUSIZ0_number = s->DataBus & 0x07;
	s->NUSIZ_M0_width = s->DataBus & 0x30;

	s->Pointer_Index_P0 = (s->Pointer_Index_P0 & 0x00003ff8) | s->NUSIZ0_number;
	s->TIA_P0_Line_Pointer = TIA_P0_Table[s->Pointer_Index_P0];

	s->Pointer_Index_M0 = (s->Pointer_Index_M0 & 0xc8) | s->NUSIZ0_number | s->NUSIZ_M0_width;
	s->TIA_M0_Line_Pointer = TIA_M0_Table[s->Pointer_Index_M0];
/*
	if(s->NUSIZ0_number == 5){
		// double and quadrouple width players need one extra offset cycle
//		if((TIA_P0_Offset) && (NUSIZ_P0_width == 0)) TIA_P0_Offset++;
		NUSIZ_P0_width = 1;
	}else if(s->NUSIZ0_number == 7){
//		if((TIA_P0_Offset) && (NUSIZ_P0_width == 0)) TIA_P0_Offset++;
		NUSIZ_P0_width = 2;
	}else{
//		if((TIA_P0_Offset) && (NUSIZ_P0_width != 0)) TIA_P0_Offset--;
		NUSIZ_P0_width = 0;
	}
*/
}


void H_NUSIZ1(struct z26_state* s){
//	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	s->NUSIZ1_number = s->DataBus & 0x07;
	s->NUSIZ_M1_width = s->DataBus & 0x30;

	s->Pointer_Index_P1 = (s->Pointer_Index_P1 & 0x00003ff8) | s->NUSIZ1_number;
	s->TIA_P1_Line_Pointer = TIA_P1_Table[s->Pointer_Index_P1];

	s->Pointer_Index_M1 = (s->Pointer_Index_M1 & 0xc8) | s->NUSIZ1_number | s->NUSIZ_M1_width;
	s->TIA_M1_Line_Pointer = TIA_M1_Table[s->Pointer_Index_M1];
/*
	if(s->NUSIZ1_number == 5){
		// double and quadrouple width players need one extra offset cycle
//		if((TIA_P1_Offset) && (NUSIZ_P1_width == 0)) TIA_P1_Offset++;
		NUSIZ_P1_width = 1;
	}else if(s->NUSIZ1_number == 7){
//		if((TIA_P1_Offset) && (NUSIZ_P1_width == 0)) TIA_P1_Offset++;
		NUSIZ_P1_width = 2;
	}else{
//		if((TIA_P1_Offset) && (NUSIZ_P1_width != 0)) TIA_P1_Offset--;
		NUSIZ_P1_width = 0;
	}
*/
}


void H_COLUP0(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_Colour_Table[P0_COLOUR] = (s->DataBus >> 1) * 257;

}


void H_COLUP1(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_Colour_Table[P1_COLOUR] = (s->DataBus >> 1) * 257;

}


void H_COLUPF(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_Colour_Table[PF_COLOUR] = (s->DataBus >> 1) * 257;

}


void H_COLUBK(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_Colour_Table[BG_COLOUR] = (s->DataBus >> 1) * 257;

}


void H_CTRLPF(struct z26_state* s){
	CatchUpPixels(s);
	if(s->DataBus & 0x01) s->CTRLPF_PF_Reflect = 1;
	else s->CTRLPF_PF_Reflect = 0;
	if(s->DataBus & 0x02) s->CTRLPF_Score = 1;
	else s->CTRLPF_Score = 0;
	if(s->DataBus & 0x04){
		s->CTRLPF_Priority = 1;
		// playfield doesn't use score colours when it has priority
		s->CTRLPF_Score = 0;
	}
	else s->CTRLPF_Priority = 0;
	s->CTRLPF_BL_width = ((s->DataBus & 0x30) >> 4);

	s->Pointer_Index_BL = (s->Pointer_Index_BL & 0x0c) | s->CTRLPF_BL_width;
	s->TIA_BL_Line_Pointer = TIA_BL_Table[s->Pointer_Index_BL];
}


void H_REFP0(struct z26_state* s){
	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	s->TIA_REFP0 = (s->DataBus & 0x08) <<9;

	s->Pointer_Index_P0 = (s->Pointer_Index_P0 & 0x00002fff) | s->TIA_REFP0;
	s->TIA_P0_Line_Pointer = TIA_P0_Table[s->Pointer_Index_P0];
}


void H_REFP1(struct z26_state* s){
	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	s->TIA_REFP1 = (s->DataBus & 0x08) << 9;

	s->Pointer_Index_P1 = (s->Pointer_Index_P1 & 0x00002fff) | s->TIA_REFP1;
	s->TIA_P1_Line_Pointer = TIA_P1_Table[s->Pointer_Index_P1];
}


void H_PF0(struct z26_state* s){
	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	s->TIA_Playfield_Bits = (s->TIA_Playfield_Bits & 0x0000ffff) | (s->DataBus << 16);
}


void H_PF1(struct z26_state* s){
	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	s->TIA_Playfield_Bits = (s->TIA_Playfield_Bits & 0x00ff00ff) | (s->DataBus << 8);
}


void H_PF2(struct z26_state* s){
	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	s->TIA_Playfield_Bits = (s->TIA_Playfield_Bits & 0x00ffff00) | s->DataBus;
}


void H_RESP0(struct z26_state* s){
	CatchUpPixels(s);
	if((s->RClock % CYCLESPERSCANLINE) < 23) s->TIA_P0_counter = 2;
	else s->TIA_P0_counter = 0;
//	if(TIA_P0_Offset) TIA_P0_Offset = 5;	// clock gets reset for sprite retriggering trick

	s->Pointer_Index_P0 = (s->Pointer_Index_P0 & 0x00003ff7);
	s->TIA_P0_Line_Pointer = TIA_P0_Table[s->Pointer_Index_P0];
}


void H_RESP1(struct z26_state* s){
	CatchUpPixels(s);
	if((s->RClock % CYCLESPERSCANLINE) < 23) s->TIA_P1_counter = 2;
	else s->TIA_P1_counter = 0;
//	if(TIA_P1_Offset) TIA_P1_Offset = 5;	// clock gets reset for sprite retriggering trick

	s->Pointer_Index_P1 = (s->Pointer_Index_P1 & 0x00003ff7);
	s->TIA_P1_Line_Pointer = TIA_P1_Table[s->Pointer_Index_P1];
}


void H_RESM0(struct z26_state* s){
	CatchUpPixels(s);
	if((s->RClock % CYCLESPERSCANLINE) < 23) s->TIA_M0_counter = 2;
	else s->TIA_M0_counter = 0;

	s->Pointer_Index_M0 = (s->Pointer_Index_M0 & 0xf7);
	s->TIA_M0_Line_Pointer = TIA_M0_Table[s->Pointer_Index_M0];
}


void H_RESM1(struct z26_state* s){
	CatchUpPixels(s);
	if((s->RClock % CYCLESPERSCANLINE) < 23) s->TIA_M1_counter = 2;
	else s->TIA_M1_counter = 0;
	s->Pointer_Index_M1 = (s->Pointer_Index_M1 & 0xf7);
	s->TIA_M1_Line_Pointer = TIA_M1_Table[s->Pointer_Index_M1];
}


void H_RESBL(struct z26_state* s){
	CatchUpPixels(s);
	if((s->RClock % CYCLESPERSCANLINE) < 23) s->TIA_BL_counter = 2;
	else s->TIA_BL_counter = 0;
}


/*
	AUDC0, AUDC1, AUDF0, AUDF1, AUDV0, AUDV1 are handled in tiasnd.c
*/


void H_GRP0(struct z26_state* s){
	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	s->TIA_GRP1_old = s->TIA_GRP1_new;
	s->TIA_GRP0_new = s->DataBus << 4;

	s->Pointer_Index_P0 = (s->Pointer_Index_P0 & 0x0000300f);
	if(s->TIA_VDELP0) s->Pointer_Index_P0 |= s->TIA_GRP0_old;
	else s->Pointer_Index_P0 |= s->TIA_GRP0_new;
	s->TIA_P0_Line_Pointer = TIA_P0_Table[s->Pointer_Index_P0];

	s->Pointer_Index_P1 = (s->Pointer_Index_P1 & 0x0000300f);
	if(s->TIA_VDELP1) s->Pointer_Index_P1 |= s->TIA_GRP1_old;
	else s->Pointer_Index_P1 |= s->TIA_GRP1_new;
	s->TIA_P1_Line_Pointer = TIA_P1_Table[s->Pointer_Index_P1];
}


void H_GRP1(struct z26_state* s){
	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	s->TIA_GRP0_old = s->TIA_GRP0_new;
	s->TIA_ENABL_old = s->TIA_ENABL_new;
	s->TIA_GRP1_new = s->DataBus << 4;

	s->Pointer_Index_P0 = (s->Pointer_Index_P0 & 0x0000300f);
	if(s->TIA_VDELP0) s->Pointer_Index_P0 |= s->TIA_GRP0_old;
	else s->Pointer_Index_P0 |= s->TIA_GRP0_new;
	s->TIA_P0_Line_Pointer = TIA_P0_Table[s->Pointer_Index_P0];

	s->Pointer_Index_P1 = (s->Pointer_Index_P1 & 0x0000300f);
	if(s->TIA_VDELP1) s->Pointer_Index_P1 |= s->TIA_GRP1_old;
	else s->Pointer_Index_P1 |= s->TIA_GRP1_new;
	s->TIA_P1_Line_Pointer = TIA_P1_Table[s->Pointer_Index_P1];

	s->Pointer_Index_BL = (s->Pointer_Index_BL & 0x0b);
	if(s->TIA_VDELBL) s->Pointer_Index_BL |= s->TIA_ENABL_old;
	else s->Pointer_Index_BL |= s->TIA_ENABL_new;
	s->TIA_BL_Line_Pointer = TIA_BL_Table[s->Pointer_Index_BL];
}


void H_ENAM0(struct z26_state* s){
	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	if(s->TIA_RESMP0) s->TIA_ENAM0 = 0;	// locked missiles are disabled
	else s->TIA_ENAM0 = (s->DataBus & 0x02) << 5;

	s->Pointer_Index_M0 = (s->Pointer_Index_M0 & 0xbf) | s->TIA_ENAM0;
	s->TIA_M0_Line_Pointer = TIA_M0_Table[s->Pointer_Index_M0];
}


void H_ENAM1(struct z26_state* s){
	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	if(s->TIA_RESMP1) s->TIA_ENAM1 = 0;	// locked missiles are disabled
	else s->TIA_ENAM1 = (s->DataBus & 0x02) << 5;

	s->Pointer_Index_M1 = (s->Pointer_Index_M1 & 0xbf) | s->TIA_ENAM1;
	s->TIA_M1_Line_Pointer = TIA_M1_Table[s->Pointer_Index_M1];
}


void H_ENABL(struct z26_state* s){
	s->TIA_Delayed_Write = 1;
	CatchUpPixels(s);
	s->TIA_Delayed_Write = 0;
	s->TIA_ENABL_new = (s->DataBus & 0x02) << 1;

	s->Pointer_Index_BL = (s->Pointer_Index_BL & 0x0b);
	if(s->TIA_VDELBL) s->Pointer_Index_BL |= s->TIA_ENABL_old;
	else s->Pointer_Index_BL |= s->TIA_ENABL_new;
	s->TIA_BL_Line_Pointer = TIA_BL_Table[s->Pointer_Index_BL];
}


void H_HMP0(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_HMP0_Value = (s->DataBus >> 4) ^ 0x08;
}


void H_HMP1(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_HMP1_Value = (s->DataBus >> 4) ^ 0x08;
}


void H_HMM0(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_HMM0_Value = (s->DataBus >> 4) ^ 0x08;
}


void H_HMM1(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_HMM1_Value = (s->DataBus >> 4) ^ 0x08;
}


void H_HMBL(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_HMBL_Value = (s->DataBus >> 4) ^ 0x08;
}


void H_VDELP0(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_VDELP0 = s->DataBus & 0x01;

	s->Pointer_Index_P0 = (s->Pointer_Index_P0 & 0x0000300f);
	if(s->TIA_VDELP0) s->Pointer_Index_P0 |= s->TIA_GRP0_old;
	else s->Pointer_Index_P0 |= s->TIA_GRP0_new;
	s->TIA_P0_Line_Pointer = TIA_P0_Table[s->Pointer_Index_P0];
}


void H_VDELP1(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_VDELP1 = s->DataBus & 0x01;

	s->Pointer_Index_P1 = (s->Pointer_Index_P1 & 0x0000300f);
	if(s->TIA_VDELP1) s->Pointer_Index_P1 |= s->TIA_GRP1_old;
	else s->Pointer_Index_P1 |= s->TIA_GRP1_new;
	s->TIA_P1_Line_Pointer = TIA_P1_Table[s->Pointer_Index_P1];
}


void H_VDELBL(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_VDELBL = s->DataBus & 0x01;

	s->Pointer_Index_BL = (s->Pointer_Index_BL & 0x0b);
	if(s->TIA_VDELBL) s->Pointer_Index_BL |= s->TIA_ENABL_old;
	else s->Pointer_Index_BL |= s->TIA_ENABL_new;
	s->TIA_BL_Line_Pointer = TIA_BL_Table[s->Pointer_Index_BL];
}


void H_RESMP0(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_RESMP0 = s->DataBus & 0x02;
	if(s->TIA_RESMP0){
		s->TIA_ENAM0 = 0;	// locking missile to player disables it
		// reset missile position to center of player position
		if(s->TIA_P0_counter < 5) s->TIA_M0_counter = s->TIA_P0_counter - 5 + 159;
		else s->TIA_M0_counter = s->TIA_P0_counter - 5;
		s->Pointer_Index_M0 = (s->Pointer_Index_M0 & 0xbf);
		s->TIA_M0_Line_Pointer = TIA_M0_Table[s->Pointer_Index_M0];
	}
}


void H_RESMP1(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_RESMP1 = s->DataBus & 0x02;
	if(s->TIA_RESMP1){
		s->TIA_ENAM1 = 0;	// locking missile to player disables it
		// reset missile position to center of player position
		if(s->TIA_P1_counter < 5) s->TIA_M1_counter = s->TIA_P1_counter - 5 + 159;
		else s->TIA_M1_counter = s->TIA_P1_counter - 5;
		s->Pointer_Index_M1 = (s->Pointer_Index_M1 & 0xbf);
		s->TIA_M1_Line_Pointer = TIA_M1_Table[s->Pointer_Index_M1];
	}
}


void H_HMOVE(struct z26_state* s){
	CatchUpPixels(s);
	if(((s->RClock % CYCLESPERSCANLINE) < 21) || ((s->RClock % CYCLESPERSCANLINE) > 74)) s->TIA_Display_HBlank = 1;
	s->TIA_HMOVE_Setup = 1;	// setting up HMOVE goes through several stages, so we only trigger it here
}


void H_HMCLR(struct z26_state* s){
	CatchUpPixels(s);
	s->TIA_HMP0_Value = 8;
	s->TIA_HMP1_Value = 8;
	s->TIA_HMM0_Value = 8;
	s->TIA_HMM1_Value = 8;
	s->TIA_HMBL_Value = 8;
}


void H_CXCLR(struct z26_state* s){
	CatchUpPixels(s);
	s->TIACollide = 0;
}

void (* TIAReadHandler[16])(struct z26_state* s) = {						
	ReadCollision,	// CXM0P
	ReadCollision,	// CXM1P
	ReadCollision,	// CXP0FB
	ReadCollision,	// CXP1FB
	ReadCollision,	// CXM0FB
	ReadCollision,	// CXM1FB
	ReadCollision,	// CXBLPF
	ReadCollision,	// CXPPMM
	ReadAnalogINPT,	// INPT0
	ReadAnalogINPT,	// INPT1
	ReadAnalogINPT,	// INPT2
	ReadAnalogINPT,	// INPT3
	ReadDigitalINPT,	// INPT4
	ReadDigitalINPT,	// INPT5
	ReadTIAdummy,	// 0x0E
	ReadTIAdummy	// 0x0F
};

void (* TIAWriteHandler[64])(struct z26_state* s) = {						
	H_VSYNC,	 //  00 -- VSYNC
	H_VBLANK,	 //  01 -- VBLANK
	H_WSYNC,	 //  02 -- WSYNC
	H_Null,		 //  03 -- reset horizontal sync
				 // 	  for factory testing only !

	H_NUSIZ0,	 //  04 -- NUSIZ0
	H_NUSIZ1,	 //  05 -- NUSIZ1
	H_COLUP0,	 //  06 -- COLUP0
	H_COLUP1,	 //  07 -- COLUP1
	H_COLUPF,	 //  08 -- COLUPF
	H_COLUBK,	 //  09 -- COLUBK
	H_CTRLPF,	 //  0a -- CTRLPF
	H_REFP0,	 //  0b -- REFP0
	H_REFP1,	 //  0c -- REFP1
	H_PF0,		 //  0d -- PF0
	H_PF1,		 //  0e -- PF1
	H_PF2,		 //  0f -- PF2
	H_RESP0,	 //  10 -- RESP0
	H_RESP1,	 //  11 -- RESP1
	H_RESM0,	 //  12 -- RESM0
	H_RESM1,	 //  13 -- RESM1
	H_RESBL,	 //  14 -- RESBL
	H_AUDC0,	 //  15 -- AUDC0
	H_AUDC1,	 //  16 -- AUDC1
	H_AUDF0,	 //  17 -- AUDF0
	H_AUDF1,	 //  18 -- AUDF1
	H_AUDV0,	 //  19 -- AUDV0
	H_AUDV1,	 //  1a -- AUDV1
	H_GRP0,		 //  1b -- GRP0
	H_GRP1,		 //  1c -- GRP1
	H_ENAM0,	 //  1d -- ENAM0
	H_ENAM1,	 //  1e -- ENAM1
	H_ENABL,	 //  1f -- ENABL
	H_HMP0,		 //  20 -- HMP0
	H_HMP1,		 //  21 -- HMP1
	H_HMM0,		 //  22 -- HMM0
	H_HMM1,		 //  23 -- HMM1
	H_HMBL,		 //  24 -- HMBL
	H_VDELP0,	 //  25 -- VDELP0
	H_VDELP1,	 //  26 -- VDELP1
	H_VDELBL,	 //  27 -- VDELBL
	H_RESMP0,	 //  28 -- RESMP0
 	H_RESMP1,	 //  29 -- RESMP1
	H_HMOVE,	 //  2a -- HMOVE
	H_HMCLR,	 //  2b -- HMCLR
	H_CXCLR,	 //  2c -- CXCLR

	H_Null,		 //  2d -- these registers are undefined
	H_Null,		 //  2e
	H_Null,		 //  2f
	H_Null,		 //  30
	H_Null,		 //  31
	H_Null,		 //  32
	H_Null,		 //  33
	H_Null,		 //  34
	H_Null,		 //  35
	H_Null,		 //  36
	H_Null,		 //  37
	H_Null,		 //  38
	H_Null,		 //  39
	H_Null,		 //  3a
	H_Null,		 //  3b
	H_Null,		 //  3c
	H_Null,		 //  3d
	H_Null,		 //  3e
	H_Null		 //  3f
};


/*
	call the right TIA write handler
*/

void C_NewTIA(struct z26_state* s){

	(* TIAWriteHandler[s->AddressBus & 0x3f])(s);
}
