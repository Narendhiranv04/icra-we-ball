(define (problem s1-mobile-move)
  (:domain kitchen-mobile-actions)

  (:objects
    fetch - robot
    home cupboard1 cupboard2 box - location
    table serving_table drawer1 drawer2 - region
    kettle coffee_jar sugar_jar spoon fork knife stirrer tongs napkin
      gso_spatula_distractor - object
    B1 D1 D2 - container
  )

  (:init
    (at fetch home)
    (base-motion-ready fetch)
    (hand-empty fetch)
    (object-at kettle home)
    (object-at coffee_jar home)
    (object-at sugar_jar home)
    (object-at spoon home)
    (object-at fork home)
    (object-at knife home)
    (object-at stirrer home)
    (object-at tongs home)
    (object-at napkin home)
    (object-at gso_spatula_distractor home)
    (graspable kettle)
    (graspable coffee_jar)
    (graspable sugar_jar)
    (graspable spoon)
    (graspable fork)
    (graspable knife)
    (graspable stirrer)
    (graspable tongs)
    (graspable napkin)
    (graspable gso_spatula_distractor)
    (container-at B1 box)
    (container-closed B1)
    (handle-graspable B1)
    (container-at D1 home)
    (container-closed D1)
    (handle-graspable D1)
    (container-at D2 home)
    (container-closed D2)
    (handle-graspable D2)

    ;; cupboard2 and box are symbolic aliases for the same right-side pose.
    (connected home cupboard1)
    (connected cupboard1 home)
    (connected home cupboard2)
    (connected cupboard2 home)
    (connected home box)
    (connected box home)
    (connected cupboard2 box)
    (connected box cupboard2)
  )

  (:goal (at fetch home))
)
