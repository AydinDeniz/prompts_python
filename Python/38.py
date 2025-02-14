
import magenta.music as mm
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
import tensorflow.compat.v1 as tf

# Load pretrained model
BUNDLE_FILE = "attention_rnn.mag"  # Replace with actual bundle file path
bundle = sequence_generator_bundle.read_bundle_file(BUNDLE_FILE)
generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
    model=mm.melody_rnn_model.MelodyRnnModel(),
    details=None,
    bundle=bundle)

# Generate a melody
def generate_melody():
    primer = mm.Melody([60])  # Middle C note
    generator_options = generator.get_generator_options()
    generator_options.generate_sections.add(
        start_time=0,
        end_time=10,
        generator_id=generator.details.id)
    
    sequence = generator.generate(primer.to_sequence(), generator_options)
    mm.sequence_proto_to_midi_file(sequence, "generated_melody.mid")
    print("Generated melody saved as 'generated_melody.mid'.")

if __name__ == "__main__":
    tf.disable_v2_behavior()  # Required for Magenta compatibility
    generate_melody()
