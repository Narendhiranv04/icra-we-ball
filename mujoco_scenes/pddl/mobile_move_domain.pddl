(define (domain kitchen-mobile-actions)
  (:requirements :strips :typing)

  (:types
    robot location object
  )

  (:predicates
    (at ?robot - robot ?location - location)
    (connected ?from - location ?to - location)
    (base-motion-ready ?robot - robot)
    (object-at ?object - object ?location - location)
    (graspable ?object - object)
    (hand-empty ?robot - robot)
    (holding ?robot - robot ?object - object)
  )

  ;; This schema defines action semantics only. The continuous trajectory is
  ;; generated and collision-checked by the Python RRT* motion planner.
  (:action move
    :parameters (?robot - robot ?from - location ?to - location)
    :precondition (and
      (at ?robot ?from)
      (connected ?from ?to)
      (base-motion-ready ?robot)
    )
    :effect (and
      (not (at ?robot ?from))
      (at ?robot ?to)
    )
  )

  ;; PDDL supplies symbolic action semantics only. Python performs the
  ;; vertical IK approach, contact-aware close, lift, and carry trajectory.
  (:action pick
    :parameters (?robot - robot ?object - object ?location - location)
    :precondition (and
      (at ?robot ?location)
      (object-at ?object ?location)
      (graspable ?object)
      (hand-empty ?robot)
    )
    :effect (and
      (not (object-at ?object ?location))
      (not (hand-empty ?robot))
      (holding ?robot ?object)
    )
  )
)
