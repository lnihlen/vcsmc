/*
	z26 RIOT emu

	I'm not really sure what mode the timer starts up in but it's not mode 1.
	Otherwise blueprnt.bin doesn't come up and others as well.
*/

#define START_TIME 0x7fff

/* <LN>
dd Timer = START_TIME;	//  the RIOT Timer
						// (gets initialized in INIT.C now)

void (* TimerReadVec)(void);	//  timer read vector

db DDR_A = 0;
db DDR_B = 0;
*/

void WriteRIOTRAM(struct z26_state* s){
	
	s->RiotRam[s->AddressBus & 0x7f] = s->DataBus;
}

void ReadRIOTRAM(struct z26_state* s){
	
	s->DataBus = s->RiotRam[s->AddressBus & 0x7f];
}

void ReadDDR_A(struct z26_state* s){			//  read data direction register A

	s->DataBus = s->DDR_A;
}

void ReadDDR_B(struct z26_state* s){			//  read data direction register B

	s->DataBus = s->DDR_B;
}

void ReadPortB(struct z26_state* s){			//  read console switches (port b)

  // <LN> Provide a sane default value.
	s->DataBus = 0x0b; // <LN> IOPortB;
}

void ReadPortA(struct z26_state* s){			//  read hand controllers (port a)
  s->DataBus = 0xff;
// <LN> 	UpdateTrakBall(ScanLine);
/* pins grounded by controller read0 even on pins where HIGH was written to (see Star Raiders) */
// <LN> 	DataBus = ((IOPortA_Controllers | IOPortA_UnusedBits) & IOPortA) &
// <LN>		((DDR_A ^ 0xff) | IOPortA_Write);
}


void WriteNothing(struct z26_state* s){}

void WritePortA(struct z26_state* s){

// <LN>	IOPortA_Write = DataBus;	// remember all written bits
	/* make sure only output bits get wtitten to SWCHA */
// <LN> 	IOPortA = (DataBus & DDR_A) | ((DDR_A ^ 0xff) & IOPortA);
	/* update controllers on SWCHA write (KEypad, Compumate, Mindlink) */
// <LN>	ControlSWCHAWrite();
}
	
void WriteDDR_A(struct z26_state* s){

	s->DDR_A = s->DataBus;
}

void WriteDDR_B(struct z26_state* s){

	s->DDR_B = s->DataBus;
}


/*
	CPU wants to read the RIOT timer
*/

void ReadTimer1(struct z26_state* s){

	s->DataBus = s->Timer & 0xff;
}

void ReadTimer8(struct z26_state* s){

	s->DataBus = (s->Timer >> 3) & 0xff;
}

void ReadTimer64(struct z26_state* s){

	s->DataBus = (s->Timer >> 6) & 0xff;
}

void ReadTimer1024(struct z26_state* s){

	s->DataBus = (s->Timer >> 10) & 0xff;
}

void ReadTimer(struct z26_state* s){

	s->Timer -= s->RCycles;	// clock this instruction
	s->RCycles = 0;		// prevent double clock
	if(s->Timer & 0x40000) s->DataBus = s->Timer & 0xff;	// timer has overflowed - switch ti TIM1T
	else (* s->TimerReadVec)(s);
}


/*
	CPU wants to read the RIOT Timer Interrupt Register
*/

void ReadTimerIntReg(struct z26_state* s){

	s->DataBus = (s->Timer >> 24) & 0x80;
/*
 I don't exactly know how many bits to leave in the Timer counter
 because I don't exactly know how long it is to the next interrupt.
 But another interrupt *does* come.  (Otherwise lockchse.bin fails.)
*/
	s->Timer &= START_TIME;
}


/*
	CPU wants to set the timer by writing to one of the RIOT timer regs:

	$294 (TIM1T)
	$295 (TIM8T)
	$296 (TIM64T)
	$297 (TIM1024T)

*/

void SetRIOTTimer1(struct z26_state* s){

	s->RCycles = 0;	// don't clock this instruction
	s->Timer = s->DataBus;
	s->TimerReadVec = &ReadTimer1;
}

void SetRIOTTimer8(struct z26_state* s){

	s->RCycles = 0;	// don't clock this instruction
	s->Timer = s->DataBus << 3;
	s->TimerReadVec = &ReadTimer8;
}

void SetRIOTTimer64(struct z26_state* s){

	s->RCycles = 0;	// don't clock this instruction
	s->Timer = s->DataBus << 6;
	s->TimerReadVec = &ReadTimer64;
}

void SetRIOTTimer1024(struct z26_state* s){

	s->RCycles = 0;	// don't clock this instruction
	s->Timer = s->DataBus << 10;
	s->TimerReadVec = &ReadTimer1024;
}


/*
	randomize RIOT timer
*/

void RandomizeRIOTTimer(struct z26_state* s) {
	/* Seconds gets set in globals.c */
	s->Timer = ((Seconds & 0xff) << 10);
}


void (* ReadRIOTTab[8])(struct z26_state* s) = {						
	ReadPortA,			//  280h PA Data
	ReadDDR_A,			//  281h PA Direction
	ReadPortB,			//  282h PB Data
	ReadDDR_B,			//  283h PB Direction
	ReadTimer,			//  284h Read Timer
	ReadTimerIntReg,	//  285h Read Timer Interrupt Register
	ReadTimer,			//  286h Read Timer
	ReadTimerIntReg		//  287h Read Timer Interrupt Register
};

void (* WriteRIOTTab[4])(struct z26_state* s) = {
	SetRIOTTimer1,		//  294h
	SetRIOTTimer8,		//  295h
	SetRIOTTimer64,		//  296h
	SetRIOTTimer1024	//  297h
};

void (* WriteRIOTTab2[4])(struct z26_state* s) = {
	WritePortA,		//  280h
	WriteDDR_A,		//  281h
	WriteNothing,	//  282h
	WriteDDR_B		//  283h
};


void Init_Riot(struct z26_state* s){
	
	s->Timer = START_TIME;
	s->TimerReadVec = &ReadTimer1024;
}


/*
	CPU wants to read a RIOT register
*/

void ReadRIOT(struct z26_state* s){

	(* ReadRIOTTab[s->AddressBus & 0x7])(s);
}

/*
	CPU wants to write to a RIOT register
*/

void WriteRIOT(struct z26_state* s){

	if(s->AddressBus & 0x10){
		if(s->AddressBus & 0x4) (* WriteRIOTTab[s->AddressBus & 0x3])(s);
		else WriteNothing(s);
	}else{
		if(!(s->AddressBus & 0x4)) (* WriteRIOTTab2[s->AddressBus & 0x3])(s);
		else WriteNothing(s);
	}
}


/*
	clock the RIOT timer (after every instruction)
	
	gets used in cpu.asm and tialine.asm
*/

void  ClockRIOT(struct z26_state* s){
	s->Timer -= s->RCycles;
}

void (* WriteRIOTHandler[32])(struct z26_state* s) = {
	WritePortA,		//  280h
	WriteDDR_A,		//  281h
	WriteNothing,	//  282h
	WriteDDR_B,		//  283h
	WriteNothing,
	WriteNothing,
	WriteNothing,
	WriteNothing,

	WritePortA,		//  280h
	WriteDDR_A,		//  281h
	WriteNothing,	//  282h
	WriteDDR_B,		//  283h
	WriteNothing,
	WriteNothing,
	WriteNothing,
	WriteNothing,

	WritePortA,		//  280h
	WriteDDR_A,		//  281h
	WriteNothing,	//  282h
	WriteDDR_B,		//  283h
	SetRIOTTimer1,		//  294h
	SetRIOTTimer8,		//  295h
	SetRIOTTimer64,		//  296h
	SetRIOTTimer1024,	//  297h

	WritePortA,		//  280h
	WriteDDR_A,		//  281h
	WriteNothing,	//  282h
	WriteDDR_B,		//  283h
	SetRIOTTimer1,		//  294h
	SetRIOTTimer8,		//  295h
	SetRIOTTimer64,		//  296h
	SetRIOTTimer1024	//  297h
};

/**
	z26 is Copyright 1997-2011 by John Saeger and contributors.  
	z26 is released subject to the terms and conditions of the 
	GNU General Public License Version 2 (GPL).  
	z26 comes with no warranty.
	Please see COPYING.TXT for details.
*/
