(define (domain kitchen-mobile-actions)
  (:requirements :strips :typing)

  (:types
    robot location object region container
  )

  (:predicates
    (at ?robot - robot ?location - location)
    (connected ?from - location ?to - location)
    (base-motion-ready ?robot - robot)
    (object-at ?object - object ?location - location)
    (graspable ?object - object)
    (hand-empty ?robot - robot)
    (holding ?robot - robot ?object - object)
    (object-in-region ?object - object ?region - region)
    (container-at ?container - container ?location - location)
    (container-closed ?container - container)
    (container-open ?container - container)
    (handle-graspable ?container - container)
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

  ;; Python samples a buffered, collision-free point inside the named region,
  ;; executes hover/descent IK, releases above the support, and retreats.
  (:action place
    :parameters (?robot - robot ?object - object ?region - region)
    :precondition (and
      (holding ?robot ?object)
    )
    :effect (and
      (not (holding ?robot ?object))
      (hand-empty ?robot)
      (object-in-region ?object ?region)
    )
  )

  ;; Python horizontally inserts the empty hand around the real handle,
  ;; confirms bilateral contact, and follows the model hinge to its limit.
  (:action open
    :parameters (?robot - robot ?container - container ?location - location)
    :precondition (and
      (at ?robot ?location)
      (container-at ?container ?location)
      (container-closed ?container)
      (handle-graspable ?container)
      (hand-empty ?robot)
    )
    :effect (and
      (not (container-closed ?container))
      (container-open ?container)
    )
  )
)
