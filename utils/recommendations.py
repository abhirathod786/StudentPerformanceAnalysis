import pandas as pd
import numpy as np

class StudentRecommendationEngine:
    """
    Generate personalized recommendations for students
    """
    
    def __init__(self, student_data):
        self.data = student_data
        self.recommendations = []
    
    def analyze_performance(self, student_row):
        """
        Analyze student performance and identify areas of improvement
        """
        analysis = {
            'math_score': student_row['math score'],
            'reading_score': student_row['reading score'],
            'writing_score': student_row['writing score'],
            'average_score': student_row['average_score'],
            'status': student_row['status'],
            'weak_subject': student_row['weak_subject'],
            'test_prep': student_row['test preparation course']
        }
        
        return analysis
    
    def get_risk_level(self, average_score):
        """
        Determine risk level based on average score
        """
        if average_score >= 70:
            return "Low Risk 🟢", "green"
        elif average_score >= 50:
            return "Medium Risk 🟡", "orange"
        else:
            return "High Risk 🔴", "red"
    
    def generate_recommendations(self, student_row):
        """
        Generate personalized recommendations for a student
        """
        recommendations = []
        analysis = self.analyze_performance(student_row)
        
        # Check test preparation
        if analysis['test_prep'] == 'none':
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Test Preparation',
                'recommendation': 'Enroll in test preparation course',
                'expected_improvement': '+8-12%',
                'reason': 'Students who complete test prep score 10% higher on average'
            })
        
        # Check weak subjects
        weak_subject = analysis['weak_subject']
        weak_score = analysis[f'{weak_subject}_score']
        
        if weak_score < 50:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': f'{weak_subject.capitalize()} Improvement',
                'recommendation': f'Focus on {weak_subject} - attend extra classes/tutoring',
                'expected_improvement': '+15-20%',
                'reason': f'Your {weak_subject} score ({weak_score}) is significantly below passing'
            })
        elif weak_score < 70:
            recommendations.append({
                'priority': 'HIGH',
                'category': f'{weak_subject.capitalize()} Enhancement',
                'recommendation': f'Practice {weak_subject} problems daily (30-45 minutes)',
                'expected_improvement': '+10-15%',
                'reason': f'Consistent practice can improve your {weak_subject} performance'
            })
        
        # Check all subjects for below 60
        for subject in ['math', 'reading', 'writing']:
            score = analysis[f'{subject}_score']
            if score < 60 and subject != weak_subject:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': f'{subject.capitalize()} Support',
                    'recommendation': f'Dedicate 20-30 minutes daily to {subject}',
                    'expected_improvement': '+8-12%',
                    'reason': f'{subject.capitalize()} score needs improvement'
                })
        
        # General study recommendations
        if analysis['average_score'] < 70:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Study Habits',
                'recommendation': 'Create a structured study schedule (2-3 hours daily)',
                'expected_improvement': '+10-15%',
                'reason': 'Consistent study routine improves overall performance'
            })
        
        # Parental education impact
        if student_row['parental level of education'] in ['some high school', 'high school']:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Additional Support',
                'recommendation': 'Seek mentorship or peer study groups',
                'expected_improvement': '+5-8%',
                'reason': 'Additional academic support can compensate for educational gaps'
            })
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def predict_improvement(self, current_score, recommendations):
        """
        Predict potential score improvement if recommendations are followed
        """
        total_improvement = 0
        
        for rec in recommendations:
            # Extract improvement percentage
            improvement_str = rec['expected_improvement']
            # Get average of range (e.g., "+8-12%" -> 10)
            numbers = [int(s) for s in improvement_str.replace('%', '').replace('+', '').split('-')]
            avg_improvement = sum(numbers) / len(numbers)
            total_improvement += avg_improvement
        
        # Cap total improvement at 30% to be realistic
        total_improvement = min(total_improvement, 30)
        
        predicted_score = min(current_score + total_improvement, 100)
        
        return predicted_score, total_improvement
    
    def get_peer_comparison(self, student_row, full_data):
        """
        Compare student with similar peers
        """
        # Filter similar students (same test prep status)
        similar_students = full_data[
            full_data['test preparation course'] == student_row['test preparation course']
        ]
        
        # Calculate statistics
        peer_avg = similar_students['average_score'].mean()
        peer_top_10 = similar_students['average_score'].quantile(0.9)
        
        student_avg = student_row['average_score']
        
        comparison = {
            'peer_average': peer_avg,
            'peer_top_10': peer_top_10,
            'student_score': student_avg,
            'gap_from_average': peer_avg - student_avg,
            'gap_from_top_10': peer_top_10 - student_avg
        }
        
        return comparison
    
    def generate_study_plan(self, weak_subject, current_scores):
        """
        Generate a weekly study plan
        """
        subjects = ['math', 'reading', 'writing']
        
        # Calculate hours based on scores
        study_hours = {}
        for subject in subjects:
            score = current_scores[f'{subject} score']
            if score < 50:
                study_hours[subject] = 3  # Critical
            elif score < 70:
                study_hours[subject] = 2  # Needs improvement
            else:
                study_hours[subject] = 1  # Maintenance
        
        # Create weekly plan
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        study_plan = {}
        
        subject_cycle = subjects * 3  # Repeat subjects
        
        for i, day in enumerate(days):
            primary_subject = subject_cycle[i % len(subject_cycle)]
            hours = study_hours[primary_subject]
            
            study_plan[day] = {
                'subject': primary_subject.capitalize(),
                'duration': f'{hours} hours',
                'focus': 'Practice problems' if primary_subject == weak_subject else 'Review concepts'
            }
        
        return study_plan
