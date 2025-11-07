@dataclass(frozen=True)
class PianoNote:
    """A container for a piano note.

    Attributes:
        number: MIDI key number.
        velocity: How hard the key was struck.
        key: Piano key corresponding to the note.
        name: Note name, in scientific pitch notation.
        fingering: Optional fingering for the note. Right hand fingers are numbered 0
            to 4, left hand 5 to 9, both starting from the thumb and ending at the
            pinky. -1 means no fingering information is available.
    """

    number: int
    velocity: int
    key: int
    name: str
    fingering: int = -1

    @staticmethod
    def create(number: int, velocity: int, fingering: int = -1) -> "PianoNote":
        """Creates a PianoNote from a MIDI pitch number and velocity."""
        if (
            not ns_constants.MIN_MIDI_VELOCITY
            <= velocity
            <= ns_constants.MAX_MIDI_VELOCITY
        ):
            raise ValueError(
                f"Velocity should be in [{ns_constants.MIN_MIDI_VELOCITY}, "
                f"{ns_constants.MAX_MIDI_VELOCITY}], got {velocity}."
            )
        if not consts.MIN_MIDI_PITCH_PIANO <= number <= consts.MAX_MIDI_PITCH_PIANO:
            raise ValueError(
                f"MIDI pitch number should be in [{consts.MIN_MIDI_PITCH_PIANO}, "
                f"{consts.MAX_MIDI_PITCH_PIANO}], got {number}."
            )
        return PianoNote(
            number=number,
            velocity=velocity,
            key=midi_number_to_key_number(number),
            name=midi_number_to_note_name(number),
            fingering=fingering,
        )
    
    
@dataclass
class NoteTrajectory:
    """A time series representation of a MIDI file.

    Attributes:
        dt: The discretization time step in seconds.
        notes: A list of lists of PianoNotes. The outer list is indexed by time step,
            and the inner list contains all the notes that are active at that time step.
        sustains: A list of integers. The i-th element indicates whether the sustain
            pedal is active at the i-th time step.
    """

    dt: float
    notes: List[List[PianoNote]]
    sustains: List[int]

    def __post_init__(self) -> None:
        """Validates the attributes."""
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if len(self.notes) != len(self.sustains):
            raise ValueError("notes and sustains must have the same length.")

    @classmethod
    def from_midi(cls, midi: MidiFile, dt: float) -> "NoteTrajectory":
        """Constructs a NoteTrajectory from a MIDI file."""
        notes, sustains = NoteTrajectory.seq_to_trajectory(midi.seq, dt)
        return cls(dt=dt, notes=notes, sustains=sustains)

    @staticmethod
    def seq_to_trajectory(
        seq: NoteSequence, dt: float
    ) -> Tuple[List[List[PianoNote]], List[int]]:
        """Converts a NoteSequence into a time series representation."""
        # Convert the note sequence into a piano roll.
        piano_roll = sequence_to_pianoroll(
            seq,
            frames_per_second=1 / dt,
            min_pitch=consts.MIN_MIDI_PITCH,
            max_pitch=consts.MAX_MIDI_PITCH,
            onset_window=0,
        )

        # Find the set of active notes at each timestep.
        notes: List[List[PianoNote]] = []
        for t, timestep in enumerate(piano_roll.active_velocities):
            notes_in_timestep: List[PianoNote] = []
            for index in np.nonzero(timestep)[0]:
                if (
                    t > 0
                    and piano_roll.active_velocities[t - 1][index]
                    and piano_roll.onset_velocities[t][index]
                ):
                    # This is to disambiguate notes that are sustained for multiple
                    # timesteps vs notes that are played consecutively over multiple
                    # timesteps.
                    continue
                velocity = int(round(timestep[index] * consts.MAX_VELOCITY))
                fingering = int(piano_roll.fingerings[t, index])
                notes_in_timestep.append(PianoNote.create(index, velocity, fingering))
            notes.append(notes_in_timestep)

        # Find the sustain pedal state at each timestep.
        sustains: List[int] = []
        prev_sustain = 0
        for timestep in piano_roll.control_changes:
            event = timestep[consts.SUSTAIN_PEDAL_CC_NUMBER]
            if 1 <= event <= consts.SUSTAIN_PEDAL_CC_NUMBER:
                sustain = 0
            elif consts.SUSTAIN_PEDAL_CC_NUMBER + 1 <= event <= consts.MAX_CC_VALUE + 1:
                sustain = 1
            else:
                sustain = prev_sustain
            sustains.append(sustain)
            prev_sustain = sustain

        return notes, sustains

    def __len__(self) -> int:
        return len(self.notes)

    def trim_silence(self) -> "NoteTrajectory":
        """Removes any leading or trailing silence from the note trajectory.

        This method modifies the note trajectory in place.
        """
        print(
            "WARNING: NoteTrajectory.trim_silence is deprecated. "
            "Trim the silence at the MIDI level instead."
        )

        # Continue removing from the front until we find a non-empty timestep.
        while len(self.notes) > 0 and len(self.notes[0]) == 0:
            self.notes.pop(0)
            self.sustains.pop(0)

        # Continue removing from the back until we find a non-empty timestep.
        while len(self.notes) > 0 and len(self.notes[-1]) == 0:
            self.notes.pop(-1)
            self.sustains.pop(-1)

        return self

    def add_initial_buffer_time(self, initial_buffer_time: float) -> "NoteTrajectory":
        """Adds artificial silence to the start of the note trajectory.

        This method modifies the note trajectory in place.
        """
        if initial_buffer_time < 0.0:
            raise ValueError("initial_buffer_time must be non-negative.")

        for _ in range(int(round(initial_buffer_time / self.dt))):
            self.notes.insert(0, [])
            self.sustains.insert(0, 0)

        return self

    def to_piano_roll(self) -> np.ndarray:
        """Returns a piano roll representation of the note trajectory.

        The piano roll is a 2D array of shape (num_timesteps, num_pitches). Each row is
        a timestep, and each column is a pitch. The value at each cell is 1 if the note
        is active at that timestep, and 0 otherwise.
        """
        frames = np.zeros((len(self.notes), consts.MAX_MIDI_PITCH), dtype=np.int32)
        for t, timestep in enumerate(self.notes):
            for note in timestep:
                frames[t, note.number] = 1
        return frames