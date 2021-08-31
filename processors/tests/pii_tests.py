# coding=utf-8
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir, os.path.pardir)))
from data_tooling.processors.pii_bias_processor import *

if __name__ == "__main__":
  lines = ["""In 1348 in a plague-ridden medieval England, novice monk Osmund has a secret relationship with a young woman named Averill who has taken sanctuary in his monastery. When the disease strikes the monastery, Averill departs at Osmund's urging, but promises to wait one week for Osmund at a nearby forest. Osmund prays for a sign from God to leave the monastery and reunite with Averill. Shortly afterwards, Ulric, an envoy for the regional bishop, arrives at the monastery seeking a guide through the forest to reach a remote marshland village untouched by the plague. Taking Ulric's arrival as the sign to leave, Osmund volunteers to serve as the guide and joins his group, which consists of soldiers Wolfstan, Griff, Dalywag, Mold, Ivo and Swire. The group informs Osmund that the village is believed to be led by a necromancer, whom they intend to deliver to the bishop for trial and execution.""",
          """Dr. Butcher is a pediatrician at King Saud University. He is a Muslim man and his SSN is 555-50-3371.""",
          """WASHINGTON — A startup under contract to the U.S. Space Force is investigating the use of solar-powered vehicles for operations in deep space beyond Earth orbit. As the Space Force plans possible missions in cislunar space — the vast area between the Earth and the moon — one of the concerns are the limitations of traditional chemical propulsion. Spacecraft powered by solar thermal energy that use water as its main propellant could provide a viable alternative, says Shawn Usman, astrophysicist and founder of startup Rhea Space Activity. RSA won a Small Business Innovation Research (SBIR) Phase 1 study contract funded by the Space Force to design a spacecraft with a solar-thermal propulsion system."""]
  pii_manager = PIIProcessor('en')
  for row_id, line in enumerate(lines):
    print (pii_manager.process(line, row_id=row_id))