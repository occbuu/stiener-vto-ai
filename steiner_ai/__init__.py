"""steiner_ai - AI-assisted Steiner-stick VTO planning.

Input = Steiner cephalometric measurements obtained with WebCeph (or other tracing software)
or measured manually by the orthodontist, plus crowding and curve-of-Spee from the dental cast.
No radiograph processing is performed.

Modules
-------
rules      deterministic Steiner-stick engine (ANB', Pog-NB', L1-NB', sticks, space table)
data       dataset loading / feature definitions
vto_model  machine-learning planner (extraction classifier + space-plan regressor + balance projection)
"""
__version__ = "1.0.0"
